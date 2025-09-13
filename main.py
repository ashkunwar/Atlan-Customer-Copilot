import os
import json
import logging
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
import uvicorn

from models import (
    Ticket, 
    TicketClassification, 
    ClassifiedTicket, 
    SingleTicketRequest, 
    BulkTicketRequest, 
    ClassificationResponse
)
from classifier import TicketClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Atlan Customer Support Copilot",
    description="AI-powered ticket classification and Response generation",
    version="1.0.0"
)

classifier = TicketClassifier()

async def rag_pipeline(question: str, topic_tags: List[str]) -> Dict:
    try:
        from enhanced_rag import EnhancedRAGPipeline
        
        rag = EnhancedRAGPipeline(groq_client=classifier.client)
        
        result = await rag.generate_answer(question, topic_tags)
        return result
        
    except ImportError as e:
        logger.warning(f"Enhanced RAG system not available: {e}")
        return await fallback_rag_pipeline(question, topic_tags)
    
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        return await fallback_rag_pipeline(question, topic_tags)

async def fallback_rag_pipeline(question: str, topic_tags: List[str]) -> Dict:
    if any(tag in ["How-to", "Product", "Best practices", "API/SDK", "SSO"] for tag in topic_tags):
        context = f"Based on Atlan documentation for topics: {', '.join(topic_tags)}"
        
        return {
            "type": "direct_answer",
            "answer": f"Based on the documentation, here's information about: {question}. {context}",
            "sources": ["https://docs.atlan.com/", "https://developer.atlan.com/"]
        }
    else:
        return {
            "type": "routing",
            "message": f"This ticket has been classified as a '{topic_tags[0] if topic_tags else 'General'}' issue and routed to the appropriate team."
        }

@app.get("/")
async def root():
    return {
        "message": "Atlan Customer Support Copilot API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/classify-single",
            "/classify-bulk", 
            "/bulk-dashboard",
            "/interactive-agent",
            "/sample-tickets"
        ]
    }

@app.post("/classify-single", response_model=ClassificationResponse)
async def classify_single_ticket(request: SingleTicketRequest):
    try:
        classification = await classifier.classify_ticket(request.ticket)
        classified_ticket = ClassifiedTicket(
            ticket=request.ticket,
            classification=classification
        )
        
        return ClassificationResponse(
            success=True,
            data=[classified_ticket],
            total_processed=1
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify-bulk", response_model=ClassificationResponse)
async def classify_bulk_tickets(request: BulkTicketRequest):
    try:
        if not request.tickets:
            raise HTTPException(status_code=400, detail="No tickets provided")
        
        classifications = await classifier.classify_tickets_bulk(request.tickets)
        
        classified_tickets = [
            ClassifiedTicket(ticket=ticket, classification=classification)
            for ticket, classification in zip(request.tickets, classifications)
        ]
        
        return ClassificationResponse(
            success=True,
            data=classified_tickets,
            total_processed=len(classified_tickets)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk classification failed: {str(e)}")

@app.get("/sample-tickets", response_model=ClassificationResponse)
async def classify_sample_tickets():
    try:
        sample_file_path = "sample_tickets.json"
        if not os.path.exists(sample_file_path):
            raise HTTPException(status_code=404, detail="Sample tickets file not found")
        
        with open(sample_file_path, "r") as f:
            tickets_data = json.load(f)
        
        tickets = [Ticket(**ticket_data) for ticket_data in tickets_data]
        
        classifications = await classifier.classify_tickets_bulk(tickets)
        
        classified_tickets = [
            ClassifiedTicket(ticket=ticket, classification=classification)
            for ticket, classification in zip(tickets, classifications)
        ]
        
        return ClassificationResponse(
            success=True,
            data=classified_tickets,
            total_processed=len(classified_tickets)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process sample tickets: {str(e)}")

@app.get("/bulk-dashboard", response_model=ClassificationResponse)
async def bulk_dashboard():
    try:
        sample_file_path = "sample_tickets.json"
        if not os.path.exists(sample_file_path):
            logger.warning(f"Sample tickets file not found: {sample_file_path}")
            return ClassificationResponse(
                success=True,
                data=[],
                total_processed=0
            )
        
        with open(sample_file_path, "r") as f:
            tickets_data = json.load(f)
        
        logger.info(f"Loaded {len(tickets_data)} sample tickets for bulk processing")
        
        tickets = [Ticket(**ticket_data) for ticket_data in tickets_data]
        
        classifications = await classifier.classify_tickets_bulk(tickets)
        
        classified_tickets = [
            ClassifiedTicket(ticket=ticket, classification=classification)
            for ticket, classification in zip(tickets, classifications)
        ]
        
        logger.info(f"Successfully classified {len(classified_tickets)} tickets for bulk dashboard")
        
        return ClassificationResponse(
            success=True,
            data=classified_tickets,
            total_processed=len(classified_tickets)
        )
    
    except Exception as e:
        logger.error(f"Failed to process bulk dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process bulk dashboard: {str(e)}")

@app.post("/upload-tickets", response_model=ClassificationResponse)
async def upload_and_classify_tickets(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        content = await file.read()
        tickets_data = json.loads(content)
        
        tickets = [Ticket(**ticket_data) for ticket_data in tickets_data]
        
        classifications = await classifier.classify_tickets_bulk(tickets)
        
        classified_tickets = [
            ClassifiedTicket(ticket=ticket, classification=classification)
            for ticket, classification in zip(tickets, classifications)
        ]
        
        return ClassificationResponse(
            success=True,
            data=classified_tickets,
            total_processed=len(classified_tickets)
        )
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded tickets: {str(e)}")

@app.post("/interactive-agent")
async def interactive_agent(
    question: str = Form(...),
    channel: str = Form("web")
):
    ticket = Ticket(id="INTERACTIVE-001", subject=question[:80], body=question)
    classification = await classifier.classify_ticket(ticket)
    topic_tags = [tag.value for tag in classification.topic_tags]
    
    analysis = {
        "topic_tags": topic_tags,
        "sentiment": classification.sentiment.value,
        "priority": classification.priority.value,
        "reasoning": classification.reasoning
    }
    
    rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
    if any(tag in rag_topics for tag in topic_tags):
        rag_result = await rag_pipeline(question, topic_tags)
        final_response = {
            "type": "direct_answer",
            "answer": rag_result.get("answer", "No answer found."),
            "sources": rag_result.get("sources", [])
        }
    else:
        final_response = {
            "type": "routing",
            "message": f"This ticket has been classified as a '{topic_tags[0]}' issue and routed to the appropriate team."
        }
    return JSONResponse({
        "internal_analysis": analysis,
        "final_response": final_response
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Atlan Customer Support Copilot"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
