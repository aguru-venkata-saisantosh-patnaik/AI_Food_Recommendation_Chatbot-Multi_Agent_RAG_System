# Cell 7: Fixed ResponseGenerator with Systematic Critical Slot Collection

import json
import logging
from enum import Enum

from utils import client

logger = logging.getLogger(__name__)


class ConversationState:
    GATHERING_INITIAL = "gathering_initial"
    NEED_DIETARY = "need_dietary"
    NEED_PRICE = "need_price"
    READY_FOR_SEARCH = "ready_for_search"
    AWAITING_SEARCH_CONFIRMATION = "awaiting_search_confirmation"
    SEARCH_READY = "search_ready"

class ResponseGenerator:
    def __init__(self, model="gpt-4o-mini", temperature=0.4):
        self.model = model
        self.temperature = temperature
        self.question_history = []
        self.conversation_state = ConversationState.GATHERING_INITIAL

    def generate(self, user_msg: str, context_summary: str, filled_slots: dict,
                question_history: list = None) -> dict:

        if question_history:
            self.question_history = question_history

        # Determine conversation state systematically
        conversation_state = self._determine_systematic_state(filled_slots, user_msg)
        self.conversation_state = conversation_state

        # Build focused prompt based on state
        prompt = self._build_systematic_prompt(user_msg, context_summary, filled_slots, conversation_state)

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a systematic food delivery assistant.
                        Follow a clear progression: gather item preferences → ask dietary along with type of cuisines → ask budget → confirm search.
                        Never repeat the same question twice. Be direct and efficient."""
                    },
                    {"role": "user", "content": prompt}
                ],
            ).choices[0].message.content.strip()

            response_data = self._parse_systematic_response(response)
            print(response_data)

            # Track questions
            if response_data.get("response_text"):
                self.question_history.append({
                    "question": response_data["response_text"],
                    "context": conversation_state,
                    "turn": len(self.question_history) + 1
                })

            response_data["question_history"] = self.question_history
            response_data["conversation_state"] = self.conversation_state

            return response_data

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._systematic_fallback_response(conversation_state, filled_slots)

    def _determine_systematic_state(self, filled_slots: dict, user_msg: str) -> str:
        """Systematically determine next required step"""

        # Check critical slots in order
        has_dietary = filled_slots.get("dietary") is not None
        has_price = filled_slots.get("price") is not None

        # Handle "no restrictions" type responses
        no_restriction_phrases = ["no restrictions", "not picky", "eat everything", "whatever", "don't care"]
        if any(phrase in user_msg.lower() for phrase in no_restriction_phrases):
            if not has_dietary:
                return ConversationState.NEED_PRICE  # Skip to price since they don't care about dietary

        # Systematic progression
        if not has_dietary:
            return ConversationState.NEED_DIETARY
        elif not has_price:
            return ConversationState.NEED_PRICE
        else:
            return ConversationState.READY_FOR_SEARCH

    def _build_systematic_prompt(self, user_msg: str, context_summary: str,
                                filled_slots: dict, conversation_state: str) -> str:
        """Build focused prompt for systematic progression"""

        base_prompt = f"""
SYSTEMATIC CONVERSATION MANAGEMENT:

Current State: {conversation_state}
User said: "{user_msg}"
Current preferences: {json.dumps(filled_slots, ensure_ascii=False)}

TASK: Generate focused response as JSON:
{{
    "response_text": "your direct response",
    "next_questions": ["follow-up question"] or [],
    "action": "ASK|SEARCH_CONFIRM|CONTINUE"
}}

RULES:
- Be direct and efficient
- Don't repeat previous questions
- Follow systematic progression
"""

        if conversation_state == ConversationState.NEED_DIETARY:
            base_prompt += """
STATE: Need dietary preference
TASK: Ask directly about dietary preference (veg/nonveg/vegan).
along with type of cuisines if applicable.
Be clear and direct. Don't ask multiple questions at once.
Set action: "ASK"
"""
        elif conversation_state == ConversationState.NEED_PRICE:
            base_prompt += """
STATE: Need price/budget information
TASK: Ask directly about budget or price range.
Be clear and specific. Ask for a number or range.
Set action: "ASK"
"""
        elif conversation_state == ConversationState.READY_FOR_SEARCH:
            base_prompt += """
STATE: Ready for search
TASK: Confirm you have enough information and ask if they want to see recommendations.
Example: "I have all the details I need! Would you like me to find [item] recommendations now?"
Set action: "SEARCH_CONFIRM"
"""
        else:
            base_prompt += """
STATE: Gathering initial preferences
TASK: Acknowledge their preference and gather basic item information.
Be conversational but focused.
Set action: "CONTINUE"
"""

        return base_prompt

    def _parse_systematic_response(self, response_text: str) -> dict:
        """Parse response with systematic validation"""

        response_text = response_text.strip()

        # Remove code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

        try:
            parsed = json.loads(response_text.strip())
            print("parsed :\n" , parsed)

            # Ensure required fields
            if "response_text" not in parsed:
                parsed["response_text"] = "I'm here to help with your food delivery!"
            if "next_questions" not in parsed:
                parsed["next_questions"] = []
            if "action" not in parsed:
                parsed["action"] = "CONTINUE"

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                "response_text": response_text,
                "next_questions": [],
                "action": "CONTINUE"
            }

    def _systematic_fallback_response(self, conversation_state: str, filled_slots: dict) -> dict:
        """Fallback responses for systematic progression"""

        if conversation_state == ConversationState.NEED_DIETARY:
            return {
                "response_text": "Do you prefer vegetarian, non-vegetarian, or vegan food?",
                "next_questions": ["Please let me know your dietary preference"],
                "action": "ASK"
            }
        elif conversation_state == ConversationState.NEED_PRICE:
            return {
                "response_text": "What's your budget for this order? Please give me a price range.",
                "next_questions": ["What's your budget in rupees?"],
                "action": "ASK"
            }
        elif conversation_state == ConversationState.READY_FOR_SEARCH:
            item = filled_slots.get("item_name", "food")
            return {
                "response_text": f"Perfect! I have all the details. Would you like me to find {item} recommendations for you now?",
                "next_questions": [],
                "action": "SEARCH_CONFIRM"
            }

        return {
            "response_text": "I'm here to help you find great food delivery options! What are you in the mood for?",
            "next_questions": ["What kind of food sounds good to you?"],
            "action": "CONTINUE"
        }

print("✅ Systematic ResponseGenerator implemented")
print("🎯 Follows clear progression: item → dietary → price → search")