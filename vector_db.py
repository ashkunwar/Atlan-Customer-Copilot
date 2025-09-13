#!/usr/bin/env python3

import json
import numpy as np
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    title: str
    content: str
    url: str
    source: str
    embedding: np.ndarray = None

class SimpleVectorDB:
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        self.model_name = model_name
        self.model = None
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.db_file = "atlan_vector_db.pkl"
        
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Using fallback TF-IDF method.")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Fallback to TF-IDF if sentence-transformers is not available"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.use_tfidf = True
            logger.info("Using TF-IDF fallback for embeddings")
        except ImportError:
            logger.error("scikit-learn not available. Using simple text matching.")
            self.use_simple_search = True
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the next 100 chars
                sentence_end = text.rfind('.', end, min(end + 100, len(text)))
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def load_knowledge_base(self, filename: str = "atlan_knowledge_base.json") -> bool:
        """Load knowledge base and create document chunks"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            logger.info(f"Loading {len(kb_data)} pages from knowledge base...")
            
            # Process each page and create document chunks
            doc_id = 0
            for page in kb_data:
                title = page.get('title', 'Untitled')
                content = page.get('content', '')
                url = page.get('url', '')
                source = page.get('source', 'unknown')
                
                if not content:
                    continue
                
                # Split content into chunks for better retrieval
                chunks = self.chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 100:  # Skip very short chunks
                        continue
                        
                    doc = Document(
                        id=f"{doc_id}_{i}",
                        title=f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                        content=chunk,
                        url=url,
                        source=source
                    )
                    self.documents.append(doc)
                
                doc_id += 1
            
            logger.info(f"Created {len(self.documents)} document chunks")
            return True
            
        except FileNotFoundError:
            logger.error(f"Knowledge base file {filename} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        if not self.documents:
            logger.error("No documents loaded")
            return
        
        if not self.model:
            self._load_embedding_model()
        
        logger.info("Creating embeddings for documents...")
        
        texts = [doc.content for doc in self.documents]
        
        if hasattr(self, 'use_tfidf') and self.use_tfidf:
            # Use TF-IDF fallback
            self.embeddings = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("Created TF-IDF embeddings")
        elif hasattr(self, 'use_simple_search'):
            # Simple keyword matching fallback
            logger.info("Using simple keyword matching")
            return
        else:
            # Use sentence transformers
            embeddings = self.model.encode(texts, show_progress_bar=True)
            self.embeddings = np.array(embeddings)
            
            # Store embeddings in documents
            for i, doc in enumerate(self.documents):
                doc.embedding = embeddings[i]
            
            logger.info(f"Created {self.embeddings.shape[0]} embeddings with dimension {self.embeddings.shape[1]}")
    
    def save_database(self):
        """Save the vector database to disk"""
        db_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model_name
        }
        
        with open(self.db_file, 'wb') as f:
            pickle.dump(db_data, f)
        
        logger.info(f"Vector database saved to {self.db_file}")
    
    def load_database(self) -> bool:
        """Load the vector database from disk"""
        try:
            with open(self.db_file, 'rb') as f:
                db_data = pickle.load(f)
            
            self.documents = db_data['documents']
            self.embeddings = db_data['embeddings']
            # Keep the current model_name (don't overwrite with old saved model)
            # This allows us to use a different model than what was saved
            saved_model = db_data.get('model_name', 'unknown')
            logger.info(f"Loaded vector database with {len(self.documents)} documents (original model: {saved_model}, using: {self.model_name})")
            
            # If the saved model is different from current, regenerate embeddings
            if saved_model != self.model_name:
                logger.warning(f"Model mismatch: saved={saved_model}, current={self.model_name}. Regenerating embeddings with new model.")
                # Force regeneration of embeddings with new model
                self._load_embedding_model()
                self.create_embeddings()
                self.save_database()  # Save with new model
                logger.info(f"Embeddings regenerated and saved with new model: {self.model_name}")
            
            return True
            
        except FileNotFoundError:
            logger.warning(f"Vector database file {self.db_file} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
            return False
    
    def simple_keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            content_words = set(doc.content.lower().split())
            title_words = set(doc.title.lower().split())
            
            # Calculate simple overlap score
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words)) * 2  # Weight title higher
            
            score = (content_overlap + title_overlap) / len(query_words)
            
            if score > 0:
                results.append((doc, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search(self, query: str, top_k: int = 5, source_filter: str = None) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        if not self.documents:
            logger.error("No documents in database")
            return []
        
        # Fallback to simple search if no embeddings
        if hasattr(self, 'use_simple_search'):
            return self.simple_keyword_search(query, top_k)
        
        # Load model if not loaded
        if not self.model and not hasattr(self, 'use_tfidf'):
            self._load_embedding_model()
        
        # Create query embedding
        if hasattr(self, 'use_tfidf') and self.use_tfidf:
            query_embedding = self.tfidf_vectorizer.transform([query])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        else:
            query_embedding = self.model.encode([query])
            # Calculate cosine similarity
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            similarities = similarities / norms
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to filter
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]
            
            # Apply source filter if specified
            if source_filter and doc.source != source_filter:
                continue
                
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_context_for_query(self, query: str, max_chars: int = 3000) -> Tuple[str, List[str]]:
        """Get relevant context for a query with source URLs"""
        
        # Determine source filter based on query content
        source_filter = None
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['api', 'sdk', 'endpoint', 'programming', 'code']):
            source_filter = 'developer'
        elif any(keyword in query_lower for keyword in ['how to', 'setup', 'configure', 'guide', 'tutorial']):
            source_filter = 'docs'
        
        # Search for relevant documents
        results = self.search(query, top_k=10, source_filter=source_filter)
        
        if not results:
            return "", []
        
        # Combine relevant content
        context_parts = []
        sources = []
        total_chars = 0
        
        for doc, score in results:
            # Only include high-relevance results
            if score < 0.1:  # Threshold for relevance
                continue
                
            content = f"Title: {doc.title}\nContent: {doc.content}\n\n"
            
            if total_chars + len(content) > max_chars:
                # Add partial content if we're near the limit
                remaining_chars = max_chars - total_chars
                if remaining_chars > 200:  # Only if we have reasonable space left
                    content = content[:remaining_chars] + "..."
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            if doc.url not in sources:
                sources.append(doc.url)
            
            total_chars += len(content)
        
        context = "".join(context_parts)
        return context, sources

def build_vector_database():
    """Build the vector database from scraped knowledge base"""
    print("🔧 Building Vector Database...")
    print("=" * 40)
    
    # Initialize vector database
    vector_db = SimpleVectorDB()
    
    # Check if database already exists
    if vector_db.load_database():
        print(f"✅ Loaded existing vector database with {len(vector_db.documents)} documents")
        response = input("Do you want to rebuild? (y/N): ").strip().lower()
        if response != 'y':
            return vector_db
    
    # Load knowledge base
    if not vector_db.load_knowledge_base():
        print("❌ Failed to load knowledge base. Run scraper first.")
        return None
    
    # Create embeddings
    print("🧮 Creating embeddings...")
    vector_db.create_embeddings()
    
    # Save database
    vector_db.save_database()
    
    print(f"✅ Vector database built successfully!")
    print(f"📊 Documents: {len(vector_db.documents)}")
    
    return vector_db

def test_search(vector_db: SimpleVectorDB):
    """Test the search functionality"""
    print("\n🔍 Testing Search Functionality...")
    print("=" * 40)
    
    test_queries = [
        "How to connect Snowflake to Atlan?",
        "API documentation for creating assets",
        "Data lineage configuration",
        "SSO setup with SAML",
        "Troubleshooting connector issues"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        context, sources = vector_db.get_context_for_query(query, max_chars=500)
        print(f"Context length: {len(context)} characters")
        print(f"Sources: {len(sources)}")
        for i, source in enumerate(sources[:3]):
            print(f"  {i+1}. {source}")

if __name__ == "__main__":
    # Build vector database
    vector_db = build_vector_database()
    
    if vector_db:
        # Test search
        test_search(vector_db)
        
        print(f"\n🎉 Vector database ready for RAG pipeline!")
    else:
        print("❌ Failed to build vector database")
