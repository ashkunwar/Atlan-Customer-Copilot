import os
import json
import asyncio
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from vector_db import SimpleVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    def __init__(self, groq_client=None):
        self.groq_client = groq_client
        self.vector_db = None
        self.knowledge_base_file = "atlan_knowledge_base.json"
        self.vector_db_file = "atlan_vector_db.pkl"
        self.initialize_vector_db()
    
    def initialize_vector_db(self):
        self.vector_db = SimpleVectorDB()
        
        if not self.vector_db.load_database():
            logger.info("No existing vector database found. Checking for knowledge base...")
            
            if Path(self.knowledge_base_file).exists():
                logger.info("Found knowledge base. Building vector database...")
                if self.vector_db.load_knowledge_base(self.knowledge_base_file):
                    self.vector_db.create_embeddings()
                    self.vector_db.save_database()
                    logger.info("Vector database built and saved")
                else:
                    logger.error("Failed to load knowledge base")
            else:
                logger.warning("No knowledge base found. RAG will use fallback responses.")
    
    def is_rag_available(self) -> bool:
        return self.vector_db is not None and len(self.vector_db.documents) > 0
    
    def should_use_rag(self, topic_tags: List[str]) -> bool:
        rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
        return any(tag in rag_topics for tag in topic_tags)
    
    def get_relevant_context(self, question: str, max_chars: int = 3000) -> Tuple[str, List[str]]:
        if not self.is_rag_available():
            return self._get_fallback_context(question), self._get_fallback_sources()
        
        try:
            context, sources = self.vector_db.get_context_for_query(question, max_chars)
            
            if not context:
                return self._get_fallback_context(question), self._get_fallback_sources()
            
            return context, sources
        
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return self._get_fallback_context(question), self._get_fallback_sources()
    
    def _get_fallback_context(self, question: str) -> str:
        question_lower = question.lower()
        
        if "snowflake" in question_lower and "connect" in question_lower:
            return """
            To connect Snowflake to Atlan:
            1. You need the following Snowflake permissions: USAGE on warehouse, database, and schema; SELECT on tables; MONITOR on warehouse
            2. Create a service account with these permissions
            3. In Atlan, go to Admin > Connectors > Add Snowflake
            4. Provide connection details: account URL, username, password, warehouse, database
            5. Test the connection and run the crawler
            
            Common issues:
            - Authentication failures: Check username/password and network access
            - Permission errors: Ensure service account has required privileges
            - Network issues: Verify Snowflake account URL and firewall settings
            """
        
        elif "api" in question_lower or "sdk" in question_lower:
            return """
            Atlan provides comprehensive APIs for programmatic access:
            
            REST API endpoints:
            - Assets API: Create, read, update assets
            - Search API: Search across the catalog
            - Lineage API: Retrieve lineage information
            - Glossary API: Manage business terms
            
            Authentication: Use API tokens (available in your profile settings)
            Base URL: https://your-tenant.atlan.com/api/meta
            
            Python SDK: pip install pyatlan
            Java SDK: Available via Maven Central
            
            Common operations:
            - Create assets: POST /entity/bulk
            - Search assets: POST /search/indexsearch
            - Get lineage: GET /lineage/{guid}
            """
        
        elif "sso" in question_lower or "saml" in question_lower:
            return """
            Setting up SSO with Atlan:
            
            SAML 2.0 Configuration:
            1. In Atlan Admin > Settings > Authentication
            2. Enable SAML SSO
            3. Configure Identity Provider details:
               - SSO URL, Entity ID, Certificate
            4. Map SAML attributes to Atlan user fields
            5. Test with a pilot user before full deployment
            
            Supported Identity Providers:
            - Okta, Azure AD, Google Workspace
            - Generic SAML 2.0 providers
            
            Troubleshooting:
            - Attribute mapping issues: Check SAML response format
            - Group assignment: Verify group claims in SAML assertions
            - Certificate errors: Ensure valid and properly formatted certificates
            """
        
        elif "lineage" in question_lower:
            return """
            Data Lineage in Atlan:
            
            Automatic lineage capture:
            - dbt: Connects via dbt Cloud or Core metadata
            - SQL-based tools: Snowflake, BigQuery, Redshift, etc.
            - ETL tools: Airflow, Fivetran, Matillion
            
            Manual lineage:
            - Use the lineage editor in the UI
            - API endpoints for programmatic lineage creation
            
            Lineage export:
            - Currently available through API calls
            - UI export features in development
            
            Troubleshooting missing lineage:
            - Check connector configuration
            - Verify SQL parsing is enabled
            - Review crawler logs for errors
            """
        
        else:
            return """
            Atlan is a modern data catalog that helps organizations:
            - Discover and understand their data assets
            - Implement data governance at scale
            - Enable self-service analytics
            - Ensure data quality and compliance
            
            Key features:
            - Automated metadata discovery
            - Data lineage visualization
            - Business glossary management
            - Data quality monitoring
            - Collaborative data stewardship
            """
    
    def _get_fallback_sources(self) -> List[str]:
        return [
            "https://docs.atlan.com/",
            "https://developer.atlan.com/",
            "https://docs.atlan.com/connectors/",
            "https://docs.atlan.com/guide/"
        ]
    
    async def generate_answer(self, question: str, topic_tags: List[str]) -> Dict:
        
        if not self.should_use_rag(topic_tags):
            return {
                "type": "routing",
                "message": f"This ticket has been classified as a '{topic_tags[0] if topic_tags else 'General'}' issue and routed to the appropriate team."
            }
        
        context, sources = self.get_relevant_context(question)
        
        if not self.groq_client:
            return {
                "type": "direct_answer",
                "answer": f"Based on the documentation, here's information about your question: {context[:500]}...",
                "sources": sources
            }
        
        try:
            response = await self._generate_llm_response(question, context, sources)
            return response
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "type": "direct_answer", 
                "answer": f"Based on the available documentation: {context[:800]}",
                "sources": sources
            }
    
    async def _generate_llm_response(self, question: str, context: str, sources: List[str]) -> Dict:
        
        prompt = f"""
You are an expert Atlan support agent. Use the provided documentation context to answer the user's question comprehensively and accurately.

User Question: {question}

Documentation Context:
{context}

Instructions:
- Provide a direct, helpful, and detailed answer
- Use the context to inform your response and be sure about it
- Be specific about steps, requirements, and configurations when applicable
- If the question is about troubleshooting, include common solutions
- If the question is about setup/configuration, provide step-by-step guidance
- Maintain a professional and helpful tone
- Only use information from the provided context
- If the context doesn't fully answer the question, acknowledge the limitation

Format your response as a comprehensive answer that directly addresses the user's question.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905", 
                messages=[
                    {"role": "system", "content": "You are an expert Atlan support agent. Provide helpful, accurate responses based on the documentation context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "type": "direct_answer",
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

def setup_rag_system():
    print("Setting up Enhanced RAG System...")
    print("=" * 45)
    
    kb_file = Path("atlan_knowledge_base.json")
    db_file = Path("atlan_vector_db.pkl")
    
    if not kb_file.exists():
        print("Knowledge base not found. Please run the scraper first")
        print(" python scraper.py")
        return False
    
    if not db_file.exists():
        print("Vector database not found. Building from knowledge base...")
        from vector_db import build_vector_database
        vector_db = build_vector_database()
        if not vector_db:
            print("Failed to build vector database")
            return False
    
    print("RAG system correct!")
    return True

async def test_rag_pipeline():
    print("\nTesting Enhanced RAG Pipeline...")
    print("=" * 40)
    
    rag = EnhancedRAGPipeline()
    
    test_questions = [
        ("How do I connect Snowflake to Atlan?", ["How-to", "Connector"]),
        ("Show me API documentation for creating assets", ["API/SDK"]),
        ("Our lineage is not showing up", ["Lineage", "Troubleshooting"]),
        ("How to configure SAML SSO?", ["SSO", "How-to"])
    ]
    
    for question, topics in test_questions:
        print(f"\nQuestion: {question}")
        print(f"Topics: {topics}")
        
        result = await rag.generate_answer(question, topics)
        
        print(f"Response Type: {result['type']}")
        if result['type'] == 'direct_answer':
            print(f"Answer Length: {len(result['answer'])} characters")
            print(f"Sources: {len(result['sources'])}")
            print(f"Answer Preview: {result['answer'][:200]}...")
        else:
            print(f"Routing: {result['message']}")

if __name__ == "__main__":
    if setup_rag_system():
        asyncio.run(test_rag_pipeline())
    else:
        print("RAG system setup failed")
