from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from enum import Enum

class SentimentEnum(str, Enum):
    FRUSTRATED = "Frustrated"
    CURIOUS = "Curious"
    ANGRY = "Angry"
    NEUTRAL = "Neutral"

class PriorityEnum(str, Enum):
    P0 = "P0 (High)"
    P1 = "P1 (Medium)"
    P2 = "P2 (Low)"

class TopicTagEnum(str, Enum):
    HOW_TO = "How-to"
    PRODUCT = "Product"
    CONNECTOR = "Connector"
    LINEAGE = "Lineage"
    API_SDK = "API/SDK"
    SSO = "SSO"
    GLOSSARY = "Glossary"
    BEST_PRACTICES = "Best practices"
    SENSITIVE_DATA = "Sensitive data"
    SECURITY = "Security"
    RBAC = "RBAC"
    AUTOMATION = "Automation"
    TROUBLESHOOTING = "Troubleshooting"
    INTEGRATION = "Integration"

class Ticket(BaseModel):
    id: str = Field(..., description="Unique ticket identifier")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Ticket body content")

class TicketClassification(BaseModel):
    topic_tags: List[TopicTagEnum] = Field(..., description="Relevant topic tags for the ticket")
    sentiment: SentimentEnum = Field(..., description="Customer sentiment")
    priority: PriorityEnum = Field(..., description="Ticket priority level")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the classification")

class ClassifiedTicket(BaseModel):
    ticket: Ticket
    classification: TicketClassification

class SingleTicketRequest(BaseModel):
    ticket: Ticket

class BulkTicketRequest(BaseModel):
    tickets: List[Ticket]

class ClassificationResponse(BaseModel):
    success: bool
    data: Optional[List[ClassifiedTicket]] = None
    error: Optional[str] = None
    total_processed: int = 0

class InteractiveAnalysis(BaseModel):
    topic_tags: List[str]
    sentiment: str
    priority: str
    reasoning: str

class DirectAnswerResponse(BaseModel):
    type: str = "direct_answer"
    answer: str
    sources: List[str] = []

class RoutingResponse(BaseModel):
    type: str = "routing"
    message: str

class InteractiveAgentResponse(BaseModel):
    internal_analysis: InteractiveAnalysis
    final_response: Dict
