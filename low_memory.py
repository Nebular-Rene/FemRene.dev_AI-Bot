import os
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from flask import Flask, request, jsonify
import torch


class DocumentationChatbot:
    def __init__(self, docs_path: str = "docs", mode: str = "fast"):
        """
        mode: 'fast' (retrieval only), 'balanced' (small LLM), 'quality' (larger LLM)
        """
        self.docs_path = docs_path
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Mode: {mode} | Device: {self.device}")
        print("Loading embedding model...")

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection("docs")
            print(f"Loaded collection: {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name="docs",
                metadata={"hnsw:space": "cosine"}
            )

        self.model = None
        self.tokenizer = None
        self.product_cache = None

    def _lazy_load_model(self):
        """Load model based on mode"""
        if self.model is None and self.mode != 'fast':
            print(f"Loading {self.mode} mode model...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.mode == 'balanced':
                # Super small and fast model
                model_name = "distilgpt2"  # Only 82MB!
                print("Using DistilGPT2 (82MB, very fast)")
            else:  # quality
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print("Using TinyLlama (4.4GB, better quality)")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded!")

    def load_documents(self):
        """Load documents efficiently"""
        print("Checking documents...")
        docs_dir = Path(self.docs_path)

        if not docs_dir.exists():
            docs_dir.mkdir(parents=True)
            return

        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Using {existing_count} cached documents")
            self._build_product_cache()
            return

        print("Processing documents...")
        all_chunks = []
        for md_file in docs_dir.rglob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                relative_path = md_file.relative_to(docs_dir)
                chunks = self._split_content(content, str(relative_path))
                all_chunks.extend(chunks)

                if len(all_chunks) >= 100:
                    self._embed_batch(all_chunks)
                    all_chunks.clear()

        if all_chunks:
            self._embed_batch(all_chunks)

        print(f"Loaded {self.collection.count()} document chunks")
        self._build_product_cache()

    def _embed_batch(self, chunks: List[Dict]):
        """Batch embed documents"""
        if not chunks:
            return

        texts = [doc['text'] for doc in chunks]
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        start_id = self.collection.count()
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{'source': doc['source']} for doc in chunks],
            ids=[f"doc_{start_id + i}" for i in range(len(texts))]
        )

    def _split_content(self, content: str, source: str) -> List[Dict]:
        """Split content into chunks"""
        chunks = []
        sections = re.split(r'\n(?=#{1,3}\s)', content)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 50:
                continue

            if len(section) > 800:
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if len(para) > 50:
                        chunks.append({'text': para, 'source': source})
            else:
                chunks.append({'text': section, 'source': source})

        return chunks

    def _build_product_cache(self):
        """Build product cache"""
        results = self.collection.get(limit=500)
        self.product_cache = set()

        for metadata in results['metadatas']:
            source_parts = metadata['source'].split('/')
            if source_parts:
                self.product_cache.add(source_parts[0].lower())

    def _extract_product_name(self, query: str) -> str:
        """Extract product from query"""
        if not self.product_cache:
            self._build_product_cache()

        query_lower = query.lower()
        for product in self.product_cache:
            if product in query_lower:
                return product
        return None

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )

        target_product = self._extract_product_name(query)

        if target_product:
            if not isinstance(target_product, list):
                target_product = [target_product]
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where={"source": {"$in": target_product}}
            )
        else:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

        return [
            {
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'rank': i + 1
            }
            for i in range(len(results['documents'][0]))
        ]

    def generate_answer_fast(self, query: str, context: List[Dict]) -> str:
        """Fast mode: return formatted context without generation"""
        if not context:
            return "No relevant documentation found for your question."

        answer = "Based on the documentation:\n\n"

        for doc in context[:3]:  # Top 3 results
            answer += f"📄 **{doc['source']}**\n"
            answer += f"{doc['text'][:300]}...\n\n"

        return answer.strip()

    def generate_answer_llm(self, query: str, context: List[Dict]) -> str:
        """Generate answer using LLM"""
        self._lazy_load_model()

        context_text = "\n".join([
            f"[{doc['source']}] {doc['text'][:300]}"
            for doc in context[:3]
        ])

        if self.mode == 'balanced':
            # Simple prompt for distilgpt2
            prompt = f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer:"
        else:
            # TinyLlama chat format
            prompt = f"""<|system|>
Answer based on the documentation provided.</s>
<|user|>
{context_text}

Question: {query}</s>
<|assistant|>
"""

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            elif "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            return answer[:500] if answer else self.generate_answer_fast(query, context)

        except Exception as e:
            print(f"Generation error: {e}")
            return self.generate_answer_fast(query, context)

    def ask(self, question: str) -> Dict:
        """Answer a question"""
        if self.collection.count() == 0:
            return {
                'answer': 'No documents loaded. Add .md files to docs/ directory.',
                'sources': []
            }

        relevant_docs = self.search_documents(question)

        if self.mode == 'fast':
            answer = self.generate_answer_fast(question, relevant_docs)
        else:
            answer = self.generate_answer_llm(question, relevant_docs)

        return {
            'answer': answer,
            'sources': list(set([doc['source'] for doc in relevant_docs])),
            'mode': self.mode
        }


# Flask API
app = Flask(__name__)
chatbot = None


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    result = chatbot.ask(question)
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'documents': chatbot.collection.count() if chatbot else 0,
        'mode': chatbot.mode if chatbot else 'unknown'
    })


@app.route('/mode/<new_mode>', methods=['POST'])
def change_mode(new_mode):
    """Change mode: fast, balanced, or quality"""
    if new_mode in ['fast', 'balanced', 'quality']:
        chatbot.mode = new_mode
        # Clear model to reload if switching to LLM mode
        if new_mode != 'fast':
            chatbot.model = None
        return jsonify({'status': 'ok', 'mode': new_mode})
    return jsonify({'error': 'Invalid mode'}), 400


if __name__ == '__main__':
    import sys

    # Choose mode from command line or default to 'fast'
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fast'

    if mode not in ['fast', 'balanced', 'quality']:
        print("Invalid mode. Choose: fast, balanced, or quality")
        sys.exit(1)

    print(f"\n🚀 Starting in '{mode}' mode")
    print("=" * 50)

    if mode == 'fast':
        print("⚡ FAST MODE - Instant responses, no LLM")
        print("   RAM: ~1-2GB | Speed: <1s")
    elif mode == 'balanced':
        print("⚖️  BALANCED MODE - DistilGPT2 (82MB)")
        print("   RAM: ~2-3GB | Speed: ~2s")
    else:
        print("🎯 QUALITY MODE - TinyLlama (4.4GB)")
        print("   RAM: ~5-7GB | Speed: ~4s")

    print("=" * 50)

    chatbot = DocumentationChatbot(mode=mode)
    chatbot.load_documents()

    print(f"\n✅ Ready! Documents: {chatbot.collection.count()}")
    print(f"🌐 Server: http://localhost:5000\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)