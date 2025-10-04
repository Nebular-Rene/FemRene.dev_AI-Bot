import os
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch


class DocumentationChatbot:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = docs_path
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        print("Loading language model (this may take a moment on first run)...")
        # Using a small, efficient model that runs on CPU
        model_name = "microsoft/phi-2"  # Small but capable model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU compatibility
            device_map="cpu"
        )
        self.model.eval()

        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")
        self.documents = []

    def load_documents(self):
        """Load all markdown files from docs directory"""
        print("Loading documents...")
        docs_dir = Path(self.docs_path)

        if not docs_dir.exists():
            print(f"Creating {self.docs_path} directory...")
            docs_dir.mkdir(parents=True)
            return

        for md_file in docs_dir.rglob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                relative_path = md_file.relative_to(docs_dir)

                # Split into chunks (by sections or paragraphs)
                chunks = self._split_content(content, str(relative_path))
                self.documents.extend(chunks)

        if self.documents:
            print(f"Loaded {len(self.documents)} document chunks")
            self._create_embeddings()
        else:
            print("No documents found. Add .md files to the docs/ directory.")

    def _split_content(self, content: str, source: str) -> List[Dict]:
        """Split markdown content into manageable chunks"""
        chunks = []

        # Split by headers or paragraphs
        sections = re.split(r'\n(?=#+\s)', content)

        for section in sections:
            if section.strip():
                # Further split if section is too long
                if len(section) > 1000:
                    paragraphs = section.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            chunks.append({
                                'text': para.strip(),
                                'source': source
                            })
                else:
                    chunks.append({
                        'text': section.strip(),
                        'source': source
                    })

        return chunks

    def _create_embeddings(self):
        """Create embeddings for all documents"""
        print("Creating embeddings...")
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.encoder.encode(texts)

        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{'source': doc['source']} for doc in self.documents],
            ids=[f"doc_{i}" for i in range(len(texts))]
        )
        print("Embeddings created successfully!")

    def _extract_product_name(self, query: str) -> str:
        """Extract product name from query by matching against known products"""
        query_lower = query.lower()

        # Get all unique product names from sources
        products = set()
        for doc in self.documents:
            source_parts = doc['source'].split('/')
            if len(source_parts) > 0:
                product_name = source_parts[0]
                products.add(product_name)

        # Check if any product name is mentioned in the query
        for product in products:
            if product.lower() in query_lower:
                return product

        return None

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents, filtering by product if mentioned"""
        query_embedding = self.encoder.encode([query])

        # Try to detect which product the user is asking about
        target_product = self._extract_product_name(query)

        if target_product:
            # Filter to only search within the specific product's docs
            where_filter = {"source": {"$contains": target_product}}
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=where_filter
            )
        else:
            # No specific product mentioned, search all docs
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

        relevant_docs = []
        for i in range(len(results['documents'][0])):
            relevant_docs.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source']
            })

        return relevant_docs

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using local Hugging Face model"""
        context_text = "\n\n".join([f"From {doc['source']}:\n{doc['text']}" for doc in context])

        prompt = f"""Documentation:
{context_text}

Question: {query}

Based on the documentation above, provide a clear and concise answer. If the information is not in the documentation, say so.

Answer:"""

        try:
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20000,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the answer part (after "Answer:")
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response[len(prompt):].strip()

            return answer if answer else "I couldn't generate a proper answer. Please rephrase your question."

        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def ask(self, question: str) -> Dict:
        """Main method to ask a question"""
        if not self.documents:
            return {
                'answer': 'No documents loaded. Please add .md files to the docs/ directory and restart.',
                'sources': []
            }

        # Search for relevant documents
        relevant_docs = self.search_documents(question)

        # Generate answer
        answer = self.generate_answer(question, relevant_docs)

        return {
            'answer': answer,
            'sources': [doc['source'] for doc in relevant_docs]
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


if __name__ == '__main__':
    print("Initializing chatbot (this may take a few minutes on first run)...")
    chatbot = DocumentationChatbot()
    chatbot.load_documents()
    app.run(debug=True, host='0.0.0.0', port=5000)