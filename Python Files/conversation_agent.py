#  Cell 9: FIXED Enhanced OpenAI Conversation Agent (Compatible Version)

from typing import Dict, Any
from datetime import datetime
import logging

from intent_classifier import OpenAIIntentClassifier
from query_enhancer import OpenAIQueryEnhancer
from response_generator import ResponseGenerator
from shards_retrieval import retrieve_all_docs_with_llm_query
from memory import history_to_json
from rerank import TwoStageContextualRerankerJSON
from utils import api_key,IntentType

logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────


class OpenAIConversationAgent:
    """
    Enhanced conversation controller with continuous flow that's compatible
    with the existing ResponseGenerator
    """

    def __init__(self, memory: 'ConversationMemory',embeddings=None):
        self.memory = memory
        self.intent_classifier = OpenAIIntentClassifier()
        self.query_enhancer = OpenAIQueryEnhancer()
        self.responder = ResponseGenerator()
        self.shard_info_path = './shard_data/shard_paths.txt'
        self.embeddings = embeddings

        # Enhanced tracking - FIXED: Initialize all attributes
        self.conversation_flow = []
        self.questions_asked = []
        self.recommendations_shown = False  # ← Now properly initialized
        self.search_history = []
        self.awaiting_search_confirm = False  # ADD THIS LINE

    def handle_turn(self, user_utterance: str, recommendations_context: dict = None) -> Dict[str, Any]:
        """Enhanced conversation turn handling with FIXED parameter compatibility"""
        try:

            # Step 2: Extract slots from user message
            context = {
                "memory": self.memory,
                "filled_slots": self.memory.get_filled_slots()
            }

            # Step 2.5: Handle slot updates based on intent type
            slots_extracted = self.query_enhancer.extract_slots_from_message(user_utterance, context)

            # 🔧 FIX: Add debugging to see what's happening
            user_intent = slots_extracted.get("user_intent", "slot_updation")
            print(f"🔍 DEBUG: Detected user_intent = '{user_intent}'")
            print(f"🔍 DEBUG: Type = {type(user_intent)}")

            slot_values = {k: v for k, v in slots_extracted.items() if k != "user_intent"}

            if user_intent == "new_query":
                print("🔄 NEW QUERY DETECTED - Clearing previous slots")
                # Clear all slots first
                self.memory.clear()
                print(f"🧹 Memory cleared: {self.memory.get_all_slots()}")

                # Replace with new values using the new method
                updated_slots = self.memory.replace_all_slots(slot_values)

                # Reset recommendations shown flag
                self.recommendations_shown = False
                print(f"🆕 New query processed - All slots replaced:")
                print(f"📋 Final slots: {self.memory.get_filled_slots()}")

            elif user_intent == "slot_updation":
                print("🔄 SLOT UPDATE - Preserving existing values, updating only mentioned slots")
                old_slots = self.memory.get_all_slots()

            # Use your existing preservation logic
                updated_slots = self.memory.update_slots_preserving_context(slot_values)

                print(f"🔄 Slot update processed:")
                print(f"📋 Previous Slots: {old_slots}")
                print(f"📋 Updated Slots: {updated_slots}")

            else:
                # Handle unexpected intents
                print(f"⚠️ Unexpected user_intent: '{user_intent}' - treating as slot_updation")
            updated_slots = self.memory.update_slots_preserving_context(slot_values)



            # Step 4: Classify intent
            intent, confidence = self.intent_classifier.classify(user_utterance, context)
            print("memory slots :", self.memory.get_all_slots() )

            # Step 5: FIXED - Generate response with correct parameters only
            response_data = self.responder.generate(
                user_msg=user_utterance,
                context_summary=self.memory.context_summary,
                filled_slots=self.memory.get_filled_slots(),
                question_history=self.questions_asked  # Only pass supported parameters
            )

            # Step 6: Update question history
            if response_data.get("question_history"):
                self.questions_asked = response_data["question_history"]

            # Step 7: Determine enhanced action
            action = self._determine_enhanced_action(
                filled_slots=self.memory.get_filled_slots(),
                response_data=response_data,
                user_message=user_utterance
            )
            print()

            # Step 8: Build complete response with continuous flow
            complete_response = {
                "action": action,
                "response": response_data.get("response_text", "I'm here to help with delivery recommendations!"),
                "next_questions": response_data.get("next_questions", []),
                "slots_updated": updated_slots,
                "all_slots": self.memory.display_all_slots(),  # Show all slots including nulls
                "intent": intent.value,
                "confidence": confidence,
                "conversation_turn": len(self.conversation_flow) + 1,
                "conversation_continues": True  # Always continue conversation
            }

        # removed search ready function from here


            # Step 10: Handle post-recommendation state
            if recommendations_context and recommendations_context.get("recommendations_shown"):
                self.recommendations_shown = True
                complete_response["post_recommendation"] = True

            # Step 11: Track conversation flow
            self.conversation_flow.append({
                "turn": len(self.conversation_flow) + 1,
                "user_input": user_utterance,
                "action": action,
                "slots_extracted": slots_extracted,
                "response_preview": response_data.get("response_text", "")[:50] + "..."
            })

            # Step 12: Update conversation history
            self.memory.add_turn(
                user_message=user_utterance,
                system_response=complete_response.get("response", ""),
                intent=intent.value,
                confidence=confidence,
                slots_updated=updated_slots
            )

            return complete_response

        except Exception as e:
            logger.error(f"Error in enhanced handle_turn: {e}")
            return self._enhanced_error_response(str(e))


    def _determine_enhanced_action(self, filled_slots: dict, response_data: dict, user_message: str) -> str:
        """Determine action with enhanced continuous flow logic"""

        # Check critical slots
        has_dietary = filled_slots.get("dietary") is not None
        has_price = filled_slots.get("price") is not None

        # Handle search confirmation responses
        if user_message.lower() in ["yes", "yeah", "sure", "ok", "find recommendations", "search"]:
            if has_dietary and has_price:
                return "SEARCH"

        # Determine based on critical slots
        if has_dietary and has_price:
            return "ASK_SEARCH_CONFIRMATION"  # Ready to ask for search confirmation
        else:
            return "ASK"  # Still need more info

    def _enhanced_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Enhanced error response that maintains conversation flow"""

        return {
            "action": "CONTINUE",
            "response": "I apologize, but I had a small hiccup. Let's continue - what would you like to explore for your meal?",
            "next_questions": ["What type of food sounds good to you right now?"],
            "slots_updated": {},
            "error": error_msg,
            "conversation_continues": True
        }

    def mark_recommendations_shown(self, recommendations_data: dict = None):
        """Mark that recommendations have been shown to user"""
        self.recommendations_shown = True
        if recommendations_data:
            from datetime import datetime
            self.search_history.append({
                "recommendations": recommendations_data,
                "timestamp": datetime.now().isoformat()
            })

    def get_enhanced_conversation_summary(self) -> dict:
        """Get enhanced conversation summary with continuous flow info"""

        return {
            "total_turns": len(self.conversation_flow),
            "questions_asked_count": len(self.questions_asked),
            "current_slots": self.memory.get_filled_slots(),
            "recommendations_shown": self.recommendations_shown,
            "search_history_count": len(self.search_history),
            "conversation_state": self.responder.conversation_state,
            "conversation_flow": self.conversation_flow[-3:],  # Last 3 turns
            "recent_questions": self.questions_asked[-3:] if self.questions_asked else []
        }

print("✅ FIXED Enhanced OpenAI Conversation Agent with parameter compatibility")
print("🔄 Compatible with existing ResponseGenerator interface")
print("💬 Ready for continuous conversation flow without parameter errors")
