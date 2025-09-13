import os
import json
from typing import List
from groq import Groq
from models import Ticket, TicketClassification, TopicTagEnum, SentimentEnum, PriorityEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable not found")
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your HF Spaces secrets or local environment.")
        
        try:
            self.client = Groq(api_key=api_key)
            self.models = [
                "moonshotai/kimi-k2-instruct-0905"
            ]
            self.model = "moonshotai/kimi-k2-instruct-0905"
            logger.info("TicketClassifier initialized successfully with Groq client")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
        
    def _create_classification_prompt(self, ticket: Ticket) -> str:
        
        topic_tags_list = [tag.value for tag in TopicTagEnum]
        sentiment_list = [sentiment.value for sentiment in SentimentEnum]
        priority_list = [priority.value for priority in PriorityEnum]
        
        prompt = f"""
You are an expert customer support analyst for Atlan, a data catalog and governance platform. 
Analyze the following support ticket and provide a classification.You give a factual response.

TICKET DETAILS:
ID: {ticket.id}
Subject: {ticket.subject}
Body: {ticket.body}

CLASSIFICATION REQUIREMENTS:

1. TOPIC TAGS (select 1-3 most relevant from the list):
{', '.join(topic_tags_list)}

2. SENTIMENT (select exactly one):
{', '.join(sentiment_list)}

3. PRIORITY (select exactly one):
{', '.join(priority_list)}

PRIORITY GUIDELINES:
- P0 (High): Urgent issues blocking customers, production failures, security concerns
- P1 (Medium): Important functionality questions, configuration issues, feature requests
- P2 (Low): General questions, documentation requests, best practices

RESPONSE FORMAT:
Please respond with a valid JSON object in this exact format:
{{
    "topic_tags": ["tag1", "tag2"],
    "sentiment": "sentiment_value",
    "priority": "priority_value", 
    "reasoning": "Brief explanation of your classification decision"
}}

IMPORTANT: Use these exact values:
- For priority: "P0 (High)", "P1 (Medium)", or "P2 (Low)"  
- For sentiment: "Frustrated", "Curious", "Angry", or "Neutral"
- For topic_tags: Use exact values from the topic list above

Ensure your response is valid JSON and uses only the exact values from the lists provided above.
"""
        return prompt

    def _normalize_topic_tags(self, tags):
        normalized_tags = []
        
        for tag in tags:
            try:
                normalized_tags.append(TopicTagEnum(tag))
            except ValueError:
                tag_lower = tag.lower()
                if 'how' in tag_lower and 'to' in tag_lower:
                    normalized_tags.append(TopicTagEnum.HOW_TO)
                elif 'api' in tag_lower or 'sdk' in tag_lower:
                    normalized_tags.append(TopicTagEnum.API_SDK)
                elif 'best' in tag_lower and 'practice' in tag_lower:
                    normalized_tags.append(TopicTagEnum.BEST_PRACTICES)
                elif 'sensitive' in tag_lower or 'pii' in tag_lower:
                    normalized_tags.append(TopicTagEnum.SENSITIVE_DATA)
                elif 'troubleshoot' in tag_lower or 'debug' in tag_lower or 'error' in tag_lower:
                    normalized_tags.append(TopicTagEnum.TROUBLESHOOTING)
                elif 'integrat' in tag_lower:
                    normalized_tags.append(TopicTagEnum.INTEGRATION)
                else:
                    normalized_tags.append(TopicTagEnum.PRODUCT)
                    logger.warning(f"Unknown topic tag '{tag}', using 'Product' as fallback")
        
        return normalized_tags or [TopicTagEnum.PRODUCT]

    def _normalize_sentiment(self, sentiment):
        try:
            return SentimentEnum(sentiment)
        except ValueError:
            sentiment_lower = sentiment.lower()
            if 'frustrat' in sentiment_lower:
                return SentimentEnum.FRUSTRATED
            elif 'angry' in sentiment_lower or 'mad' in sentiment_lower:
                return SentimentEnum.ANGRY
            elif 'curious' in sentiment_lower or 'interest' in sentiment_lower:
                return SentimentEnum.CURIOUS
            else:
                return SentimentEnum.NEUTRAL

    def _normalize_priority(self, priority):
        try:
            return PriorityEnum(priority)
        except ValueError:
            priority_lower = str(priority).lower()
            if 'p0' in priority_lower or 'high' in priority_lower or 'urgent' in priority_lower:
                return PriorityEnum.P0
            elif 'p2' in priority_lower or 'low' in priority_lower:
                return PriorityEnum.P2
            else:
                return PriorityEnum.P1

    async def classify_ticket(self, ticket: Ticket) -> TicketClassification:
        for model in self.models:
            try:
                prompt = self._create_classification_prompt(ticket)
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert customer support analyst. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                logger.info(f"Raw AI response for ticket {ticket.id} using model {model}: {content}")
                
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                try:
                    classification_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for ticket {ticket.id} using model {model}: {e}")
                    continue
                
                topic_tags = self._normalize_topic_tags(classification_data.get("topic_tags", ["Product"]))
                sentiment = self._normalize_sentiment(classification_data.get("sentiment", "Neutral"))
                priority = self._normalize_priority(classification_data.get("priority", "P1"))
                
                return TicketClassification(
                    topic_tags=topic_tags,
                    sentiment=sentiment,
                    priority=priority,
                    reasoning=classification_data.get("reasoning", f"AI-generated classification using {model}")
                )
                
            except Exception as e:
                logger.error(f"Error classifying ticket {ticket.id} with model {model}: {str(e)}")
                continue
        
        logger.error(f"All models failed for ticket {ticket.id}, using fallback")
        return TicketClassification(
            topic_tags=[TopicTagEnum.PRODUCT],
            sentiment=SentimentEnum.NEUTRAL,
            priority=PriorityEnum.P1,
            reasoning="All AI models failed, using fallback classification"
        )

    async def classify_tickets_bulk(self, tickets: List[Ticket]) -> List[TicketClassification]:
        classifications = []
        
        for ticket in tickets:
            classification = await self.classify_ticket(ticket)
            classifications.append(classification)
            logger.info(f"Classified ticket {ticket.id}")
        
        return classifications
