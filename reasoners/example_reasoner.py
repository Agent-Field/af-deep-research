"""
Example Reasoner - deep-research-agent

This file demonstrates how to create reusable reasoner functions using the new decorator pattern.
Reasoners are AI-powered functions that perform reasoning, analysis, or decision-making tasks.

NOTE: This template is for reference only. In the new decorator-based approach,
reasoners are defined directly in main.py using the @app.reasoner() decorator.

Created: 2025-07-09 16:20:06 EDT
Author: Agent Field Team
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Example Pydantic models for structured output
class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

class AnalysisResult(BaseModel):
    category: str
    summary: str
    confidence: float
    key_points: list[str]

# RECOMMENDED: Use the decorator pattern in main.py instead:
#
# @app.reasoner()
# async def analyze_sentiment(text: str) -> SentimentResult:
#     """Analyzes text sentiment using AI."""
#     result = await app.ai(
#         system="You are a sentiment analysis expert. Provide a concise reasoning.",
#         user=f"Analyze the sentiment of: '{text}'",
#         schema=SentimentResult,
#     )
#     return result
#
# @app.reasoner()
# async def analyze_content(content: str) -> AnalysisResult:
#     """Analyzes content and provides structured insights."""
#     result = await app.ai(
#         "Analyze this content and provide structured analysis",
#         content,
#         schema=AnalysisResult,
#     )
#     return result
#
# @app.reasoner()
# async def multimodal_analysis(
#     text: str, image_url: str = None, audio_file: str = None
# ) -> dict:
#     """Analyzes multimodal content (text, image, audio)."""
#     inputs = [
#         "Analyze the content considering all provided media:",
#         text,
#     ]
#     
#     if image_url:
#         inputs.append(image_url)  # Auto-detected as image
#     if audio_file:
#         inputs.append(audio_file)  # Auto-detected as audio
#     
#     response = await app.ai(
#         *inputs,
#         model="gpt-4o",  # Use vision-capable model
#         temperature=0.2,
#         max_tokens=1000,
#     )
#     return {"analysis": response}

# Legacy examples below (for reference only)

async def example_reasoning_task(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Example reasoning function using old manual registration approach.
    
    For new projects, define reasoners directly in main.py using @app.reasoner() decorator.
    This provides automatic schema generation and better integration with the Brain system.
    
    Args:
        query (str): The input query to analyze
        context (Optional[Dict[str, Any]]): Additional context for reasoning
        
    Returns:
        Dict[str, Any]: Reasoning results with confidence score
    """
    
    logger.warning("Using deprecated manual reasoner registration. Consider migrating to @app.reasoner() decorator pattern.")
    logger.info(f"Processing reasoning task for query: {query[:50]}...")
    
    # Example reasoning logic (replace with your implementation)
    if not query or not query.strip():
        return {
            "result": "Empty query provided",
            "confidence": 0.0,
            "error": "No input to analyze",
            "migration_note": "Consider using @app.reasoner() decorator in main.py with app.ai() for AI-powered analysis"
        }
    
    # Simple keyword-based analysis (replace with actual AI reasoning using app.ai())
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    confidence = min(len(keywords) * 0.2, 1.0)
    
    # Example context usage
    context_boost = 0.0
    if context and "user_history" in context:
        context_boost = 0.1
        
    final_confidence = min(confidence + context_boost, 1.0)
    
    return {
        "result": f"Analyzed query with {len(keywords)} significant keywords",
        "confidence": final_confidence,
        "keywords": keywords,
        "reasoning_type": "keyword_analysis",
        "context_used": context is not None,
        "processed_at": "2025-07-09 16:20:06 EDT",
        "migration_note": "Consider using @app.reasoner() decorator in main.py with app.ai() for AI-powered analysis"
    }

async def sentiment_analyzer(text: str) -> Dict[str, Any]:
    """
    DEPRECATED: Example sentiment analysis reasoner using old approach.
    
    For new projects, define this in main.py using @app.reasoner() decorator
    and leverage app.ai() for AI-powered sentiment analysis.
    
    Example of modern approach:
    
    @app.reasoner()
    async def analyze_sentiment(text: str) -> SentimentResult:
        result = await app.ai(
            system="You are a sentiment analysis expert.",
            user=f"Analyze sentiment of: '{text}'",
            schema=SentimentResult,
        )
        return result
    
    Args:
        text (str): Text to analyze for sentiment
        
    Returns:
        Dict[str, Any]: Sentiment analysis results
    """
    
    logger.warning("Using deprecated manual reasoner registration. Consider migrating to @app.reasoner() decorator pattern.")
    logger.info("Performing sentiment analysis...")
    
    # Simple sentiment analysis (replace with app.ai() for better results)
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "disappointing"]
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(positive_count / len(words) * 2, 1.0)
    elif negative_count > positive_count:
        sentiment = "negative" 
        score = min(negative_count / len(words) * 2, 1.0)
    else:
        sentiment = "neutral"
        score = 0.5
        
    return {
        "sentiment": sentiment,
        "score": score,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "confidence": min(abs(positive_count - negative_count) / len(words) * 2, 1.0),
        "migration_note": "Consider using @app.reasoner() decorator in main.py with app.ai() for AI-powered analysis"
    }

# Additional examples for reference

async def content_classifier(content: str, categories: list[str]) -> Dict[str, Any]:
    """
    DEPRECATED: Example content classification.
    
    Modern approach using app.ai():
    
    @app.reasoner()
    async def classify_content(content: str, categories: list[str]) -> dict:
        result = await app.ai(
            system=f"Classify content into one of these categories: {', '.join(categories)}",
            user=f"Classify this content: {content}",
            temperature=0.1,  # Low temperature for consistent classification
        )
        return {"classification": result, "categories": categories}
    """
    
    # Simple classification logic (replace with app.ai())
    content_lower = content.lower()
    scores = {}
    
    for category in categories:
        # Simple keyword matching (replace with AI)
        category_keywords = category.lower().split()
        score = sum(1 for keyword in category_keywords if keyword in content_lower)
        scores[category] = score / len(category_keywords) if category_keywords else 0
    
    best_category = max(scores, key=scores.get) if scores else "unknown"
    
    return {
        "classification": best_category,
        "scores": scores,
        "confidence": scores.get(best_category, 0),
        "migration_note": "Use @app.reasoner() with app.ai() for better classification"
    }

async def question_answerer(question: str, context: str) -> Dict[str, Any]:
    """
    DEPRECATED: Example question answering.
    
    Modern approach using app.ai():
    
    @app.reasoner()
    async def answer_question(question: str, context: str) -> dict:
        result = await app.ai(
            system="Answer the question based on the provided context.",
            user=f"Context: {context}\n\nQuestion: {question}",
            temperature=0.3,
        )
        return {"answer": result, "question": question}
    """
    
    # Simple keyword-based answering (replace with app.ai())
    question_words = set(question.lower().split())
    context_words = set(context.lower().split())
    
    overlap = len(question_words.intersection(context_words))
    relevance = overlap / len(question_words) if question_words else 0
    
    if relevance > 0.3:
        answer = f"Based on the context, the answer relates to the overlapping concepts: {', '.join(question_words.intersection(context_words))}"
    else:
        answer = "The context doesn't seem to contain relevant information for this question."
    
    return {
        "answer": answer,
        "relevance": relevance,
        "question": question,
        "migration_note": "Use @app.reasoner() with app.ai() for better question answering"
    }
