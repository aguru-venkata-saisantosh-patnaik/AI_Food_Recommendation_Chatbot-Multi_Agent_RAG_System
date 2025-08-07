# Intent Classification

from enum import Enum
from typing import Dict, Optional, Tuple
import logging

from utils import call_openai,IntentType, rate_limiter

logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────

class OpenAIIntentClassifier:
    """Intent classifier using OpenAI"""

    def __init__(self):
        self.fallback_patterns = {
            IntentType.RECOMMEND: ["recommend", "suggest", "looking for", "show me", "want", "find"],
            IntentType.FILTER_UPDATE: ["budget", "veg", "nonveg", "vegan", "cheap", "spicy", "sweet", "cuisine"],
            IntentType.GREETING: ["hi", "hello", "hey", "good morning", "good afternoon"],
            IntentType.GOODBYE: ["bye", "thanks", "thank you", "quit", "done", "goodbye"],
            IntentType.FEEDBACK: ["good", "great", "bad", "love", "hate", "like", "dislike"],
        }

    def classify(self, utterance: str, context: Optional[Dict] = None) -> Tuple[IntentType, float]:
        """Classify intent using OpenAI with fallback"""
        try:
            rate_limiter.wait_if_needed()

            prompt = f'''
You are an expert at understanding user intents in food recommendation conversations.
Classify the user's message into one of these intents:
- RECOMMEND: User wants food recommendations
- FILTER_UPDATE: User is specifying preferences (dietary, cuisine, price, etc.)
- CLARIFICATION: User needs clarification or has questions
- FEEDBACK: User is giving feedback on recommendations
- GREETING: User is starting the conversation
- GOODBYE: User is ending the conversation
- OTHER: Unclear or unrelated intent

Respond with ONLY the intent name and confidence score (0-1) in this format:
INTENT: [intent_name]
CONFIDENCE: [0-1]

User message: "{utterance}"
Classify this message:'''

            response_text = call_openai(prompt)
            print(response_text)
            return self._parse_intent_response(response_text)

        except Exception as e:
            logger.warning(f"OpenAI intent classification failed: {e}, using fallback")
            return self._fallback_classification(utterance)

    def _parse_intent_response(self, response_text: str) -> Tuple[IntentType, float]:
        """Parse OpenAI response for intent classification"""
        try:
            lines = response_text.split('\n')
            intent_line = next((line for line in lines if line.startswith("INTENT:")), "")
            confidence_line = next((line for line in lines if line.startswith("CONFIDENCE:")), "")

            if intent_line and confidence_line:
                intent_str = intent_line.split(":", 1)[1].strip().upper()
                confidence_str = confidence_line.split(":", 1)[1].strip()

                # Map string to enum
                intent_mapping = {
                    "RECOMMEND": IntentType.RECOMMEND,
                    "FILTER_UPDATE": IntentType.FILTER_UPDATE,
                    "CLARIFICATION": IntentType.CLARIFICATION,
                    "FEEDBACK": IntentType.FEEDBACK,
                    "GREETING": IntentType.GREETING,
                    "GOODBYE": IntentType.GOODBYE,
                    "OTHER": IntentType.OTHER
                }

                intent = intent_mapping.get(intent_str, IntentType.OTHER)
                confidence = float(confidence_str)

                return intent, min(max(confidence, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Failed to parse intent response: {e}")

        return self._fallback_classification(response_text)

    def _fallback_classification(self, utterance: str) -> Tuple[IntentType, float]:
        """Fallback classification using keyword matching"""
        utterance_lower = utterance.lower()

        for intent_type, keywords in self.fallback_patterns.items():
            if any(keyword in utterance_lower for keyword in keywords):
                return intent_type, 0.6  # Medium confidence for fallback

        return IntentType.OTHER, 0.3

