import streamlit as st
st.set_page_config(
    page_title="🎯 Atlan Customer Support Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
import asyncio
import logging
import os
from typing import List, Dict
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_api_key():
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return api_key
    
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            return st.secrets['GROQ_API_KEY']
    except Exception as e:
        logger.warning(f"Could not access Streamlit secrets: {e}")
    
    return None

groq_api_key = get_groq_api_key()

if not groq_api_key:
    st.error("🚨 **GROQ API Key Missing!**")
    st.markdown("""
    ### Please add your GROQ API key:
    
    **For Hugging Face Spaces:**
    1. Go to your Space settings tab
    2. Scroll to "Repository secrets" 
    3. Add secret: `GROQ_API_KEY` = `your_actual_key`
    4. Restart the space
    
    **For local development:**
    Add to `.streamlit/secrets.toml`:
    ```toml
    GROQ_API_KEY = "your_groq_api_key_here"
    ```
    
    **Get your API key:** https://console.groq.com/keys
    """)
    st.stop()
else:
    os.environ['GROQ_API_KEY'] = groq_api_key
    st.success("🔑 API key loaded successfully")

try:
    from models import Ticket, TicketClassification, TopicTagEnum, SentimentEnum, PriorityEnum
    from classifier import TicketClassifier
    from enhanced_rag import EnhancedRAGPipeline
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.error("Please ensure all required files are present in the directory")
    st.stop()

st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .ticket-card {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tag {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_ai_models():
    try:
        classifier = TicketClassifier()
        rag_pipeline = EnhancedRAGPipeline(groq_client=classifier.client)
        return classifier, rag_pipeline
    except Exception as e:
        st.error(f"❌ Failed to initialize AI models: {e}")
        return None, None

def load_sample_tickets():
    try:
        with open('sample_tickets.json', 'r') as f:
            tickets_data = json.load(f)
        return [Ticket(**ticket_data) for ticket_data in tickets_data]
    except FileNotFoundError:
        st.warning("📋 Sample tickets file not found. Using demo data for cloud deployment.")
        demo_tickets = [
            {
                "id": "DEMO-001",
                "subject": "Demo ticket - Connection issue",
                "body": "This is a demo ticket showing connection problems with our data source."
            },
            {
                "id": "DEMO-002", 
                "subject": "Demo ticket - API question",
                "body": "This is a demo ticket asking about API usage and documentation."
            }
        ]
        return [Ticket(**ticket_data) for ticket_data in demo_tickets]
    except Exception as e:
        st.error(f"❌ Error loading tickets: {e}")
        return []

async def classify_tickets_async(classifier, tickets):
    try:
        classifications = await classifier.classify_tickets_bulk(tickets)
        return list(zip(tickets, classifications))
    except Exception as e:
        st.error(f"❌ Classification error: {e}")
        return []

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def calculate_stats(classified_tickets):
    if not classified_tickets:
        return {
            'total': 0,
            'high_priority': 0,
            'frustrated': 0,
            'rag_eligible': 0,
            'most_common_tag': 'N/A',
            'tag_counts': {}
        }
    
    total = len(classified_tickets)
    high_priority = sum(1 for _, classification in classified_tickets 
                       if classification.priority == PriorityEnum.P0)
    frustrated = sum(1 for _, classification in classified_tickets 
                    if classification.sentiment in [SentimentEnum.FRUSTRATED, SentimentEnum.ANGRY])
    
    rag_topics = ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
    rag_eligible = sum(1 for _, classification in classified_tickets 
                      if any(tag.value in rag_topics for tag in classification.topic_tags))
    
    tag_counts = {}
    for _, classification in classified_tickets:
        for tag in classification.topic_tags:
            tag_counts[tag.value] = tag_counts.get(tag.value, 0) + 1
    
    most_common_tag = max(tag_counts.keys(), key=lambda x: tag_counts[x]) if tag_counts else 'N/A'
    
    return {
        'total': total,
        'high_priority': high_priority,
        'frustrated': frustrated,
        'rag_eligible': rag_eligible,
        'most_common_tag': most_common_tag,
        'tag_counts': tag_counts
    }

def display_ticket_card(ticket, classification):
    with st.container():
        st.markdown(f"**{ticket.id}**")
        st.write(f"**Subject:** {ticket.subject}")
        st.write(f"**Message:** {ticket.body[:300]}{'...' if len(ticket.body) > 300 else ''}")
        
        st.write("**📋 Topics:**")
        cols = st.columns(len(classification.topic_tags))
        for i, tag in enumerate(classification.topic_tags):
            with cols[i]:
                st.markdown(f'<span style="background: #667eea; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem; margin: 0.1rem;">{tag.value}</span>', unsafe_allow_html=True)
        
        sentiment_color = '#ff6b6b' if 'frustrated' in classification.sentiment.value.lower() else '#ff3838' if 'angry' in classification.sentiment.value.lower() else '#4ecdc4' if 'curious' in classification.sentiment.value.lower() else '#95a5a6'
        st.markdown(f"**😊 Sentiment:** <span style='background: {sentiment_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem;'>{classification.sentiment.value}</span>", unsafe_allow_html=True)
        
        priority_color = '#ff3838' if 'P0' in classification.priority.value else '#ffa726' if 'P1' in classification.priority.value else '#66bb6a'
        st.markdown(f"**🔥 Priority:** <span style='background: {priority_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem;'>{classification.priority.value}</span>", unsafe_allow_html=True)
        
        st.write(f"**🤖 AI Reasoning:** {classification.reasoning}")
        st.divider()

def main():
    classifier, rag_pipeline = initialize_ai_models()
    
    if classifier is None or rag_pipeline is None:
        st.stop()
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Atlan Customer Support Copilot</h1>
        <p>AI-powered ticket classification and intelligent response generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Bulk Classification Dashboard", 
        "🤖 Interactive AI Agent",
        "📝 Single Ticket Classification",
        "📂 Upload & Classify"
    ])
    
    # Page routing
    if page == "📊 Bulk Classification Dashboard":
        bulk_dashboard_page(classifier)
    elif page == "🤖 Interactive AI Agent":
        interactive_agent_page(classifier, rag_pipeline)
    elif page == "📝 Single Ticket Classification":
        single_ticket_page(classifier)
    elif page == "📂 Upload & Classify":
        upload_classify_page(classifier)

def bulk_dashboard_page(classifier):
    st.header("📊 Bulk Classification Dashboard")
    st.subheader("Auto-loaded sample tickets with AI classification")
    
    if 'bulk_results' not in st.session_state:
        st.session_state.bulk_results = None
    
    if st.session_state.bulk_results is None:
        with st.spinner("🔄 Loading and classifying sample tickets..."):
            tickets = load_sample_tickets()
            if tickets:
                try:
                    classified_tickets = run_async(classify_tickets_async(classifier, tickets))
                    st.session_state.bulk_results = classified_tickets
                    st.success(f"✅ Successfully classified {len(classified_tickets)} tickets!")
                except Exception as e:
                    st.error(f"❌ Error during classification: {e}")
                    st.session_state.bulk_results = []
            else:
                st.session_state.bulk_results = []
    
    if st.session_state.bulk_results:
        # Display statistics
        stats = calculate_stats(st.session_state.bulk_results)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📋 Total Tickets", stats['total'])
        with col2:
            st.metric("🚨 High Priority", stats['high_priority'])
        with col3:
            st.metric("😤 Frustrated/Angry", stats['frustrated'])
        with col4:
            st.metric("🤖 RAG-Eligible", stats['rag_eligible'])
        with col5:
            st.metric("🏷️ Top Topic", stats['most_common_tag'])
        
        if stats['tag_counts']:
            col1, col2 = st.columns(2)
            
            with col1:
                priority_data = {}
                for _, classification in st.session_state.bulk_results:
                    priority = classification.priority.value
                    priority_data[priority] = priority_data.get(priority, 0) + 1
                
                fig_priority = px.pie(
                    values=list(priority_data.values()),
                    names=list(priority_data.keys()),
                    title="📊 Priority Distribution",
                    color_discrete_map={
                        'P0 (High)': '#ff3838',
                        'P1 (Medium)': '#ffa726',
                        'P2 (Low)': '#66bb6a'
                    }
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            with col2:
                fig_tags = px.bar(
                    x=list(stats['tag_counts'].values()),
                    y=list(stats['tag_counts'].keys()),
                    orientation='h',
                    title="🏷️ Topic Distribution",
                    labels={'x': 'Count', 'y': 'Topics'}
                )
                fig_tags.update_layout(height=400)
                st.plotly_chart(fig_tags, use_container_width=True)
        
        st.subheader("📋 All Classified Tickets")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            priority_filter = st.selectbox("Filter by Priority", 
                ["All"] + [p.value for p in PriorityEnum])
        with col2:
            sentiment_filter = st.selectbox("Filter by Sentiment", 
                ["All"] + [s.value for s in SentimentEnum])
        with col3:
            topic_filter = st.selectbox("Filter by Topic", 
                ["All"] + [t.value for t in TopicTagEnum])
        
        filtered_results = st.session_state.bulk_results
        if priority_filter != "All":
            filtered_results = [(t, c) for t, c in filtered_results if c.priority.value == priority_filter]
        if sentiment_filter != "All":
            filtered_results = [(t, c) for t, c in filtered_results if c.sentiment.value == sentiment_filter]
        if topic_filter != "All":
            filtered_results = [(t, c) for t, c in filtered_results if any(tag.value == topic_filter for tag in c.topic_tags)]
        
        st.info(f"Showing {len(filtered_results)} of {len(st.session_state.bulk_results)} tickets")
        
        for ticket, classification in filtered_results:
            display_ticket_card(ticket, classification)
    
    if st.button("🔄 Refresh Classifications"):
        st.session_state.bulk_results = None
        st.rerun()

def interactive_agent_page(classifier, rag_pipeline):
    st.header("🤖 Interactive AI Agent")
    st.subheader("Submit a new ticket or question from any channel")
    
    with st.form("interactive_form"):
        question = st.text_area(
            "Customer Question or Ticket:",
            placeholder="Enter the customer's question or ticket description...",
            height=150
        )
        
        channel = st.selectbox(
            "Channel:",
            ["Web", "Email", "WhatsApp", "Voice", "Live Chat"]
        )
        
        submit_button = st.form_submit_button("🚀 Process with AI Agent")
    
    if submit_button and question:
        with st.spinner("🤖 Analyzing question and generating response..."):
            try:
                ticket = Ticket(id="INTERACTIVE-001", subject=question[:80], body=question)
                
                classification = run_async(classifier.classify_ticket(ticket))
                topic_tags = [tag.value for tag in classification.topic_tags]
                
                rag_result = run_async(rag_pipeline.generate_answer(question, topic_tags))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Internal Analysis (Back-end View)")
                    
                    st.markdown(f"""
                    **🏷️ Topic Tags:** {', '.join([f'`{tag}`' for tag in topic_tags])}
                    
                    **😊 Sentiment:** `{classification.sentiment.value}`
                    
                    **⚡ Priority:** `{classification.priority.value}`
                    
                    **🤖 AI Reasoning:** {classification.reasoning}
                    """)
                
                with col2:
                    st.subheader("💬 Final Response (Front-end View)")
                    
                    if rag_result['type'] == 'direct_answer':
                        st.success("💡 Direct Answer (RAG-Generated)")
                        st.write(rag_result['answer'])
                        
                        if rag_result.get('sources'):
                            st.subheader("📚 Sources:")
                            for source in rag_result['sources']:
                                st.markdown(f"- [{source}]({source})")
                    else:
                        st.warning("📋 Ticket Routed")
                        st.write(rag_result['message'])
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")

def single_ticket_page(classifier):
    st.header("📝 Single Ticket Classification")
    
    with st.form("single_ticket_form"):
        ticket_id = st.text_input("Ticket ID:", placeholder="e.g., TICKET-001")
        subject = st.text_input("Subject:", placeholder="Enter ticket subject")
        body = st.text_area("Message Body:", placeholder="Enter the full ticket message...", height=150)
        
        classify_button = st.form_submit_button("🔍 Classify Ticket")
    
    if classify_button and ticket_id and subject and body:
        with st.spinner("🔄 Classifying ticket..."):
            try:
                ticket = Ticket(id=ticket_id, subject=subject, body=body)
                classification = run_async(classifier.classify_ticket(ticket))
                
                st.success("✅ Classification complete!")
                display_ticket_card(ticket, classification)
                
            except Exception as e:
                st.error(f"❌ Error classifying ticket: {e}")

def upload_classify_page(classifier):
    st.header("📂 Upload & Classify Tickets")
    
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            tickets_data = json.load(uploaded_file)
            tickets = [Ticket(**ticket_data) for ticket_data in tickets_data]
            
            st.info(f"📄 Loaded {len(tickets)} tickets from file")
            
            if st.button("🚀 Classify All Tickets"):
                with st.spinner("🔄 Classifying tickets..."):
                    try:
                        classified_tickets = run_async(classify_tickets_async(classifier, tickets))
                        
                        st.success(f"✅ Successfully classified {len(classified_tickets)} tickets!")
                        
                        stats = calculate_stats(classified_tickets)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", stats['total'])
                        with col2:
                            st.metric("High Priority", stats['high_priority'])
                        with col3:
                            st.metric("Frustrated", stats['frustrated'])
                        with col4:
                            st.metric("RAG-Eligible", stats['rag_eligible'])
                        
                        for ticket, classification in classified_tickets:
                            display_ticket_card(ticket, classification)
                            
                    except Exception as e:
                        st.error(f"❌ Error during classification: {e}")
                        
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎯 <strong>Atlan Customer Support Copilot</strong> - AI-powered ticket classification and response generation</p>
        <p>Built with Streamlit • Powered by Groq AI • Enhanced RAG Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
