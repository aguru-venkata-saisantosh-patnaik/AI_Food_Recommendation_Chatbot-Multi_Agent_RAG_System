# Cell 4: ConversationMemory Class (from conversation_gemini.ipynb)

from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from utils import REQUIRED_SLOTS, DialogueState

logger = logging.getLogger(__name__)



@dataclass
class ConversationTurn:
    """Single conversation turn data"""
    timestamp: str
    user_message: str
    system_response: str
    intent: str
    confidence: float
    slots_updated: Dict[str, Any] = field(default_factory=dict)
    action_state : str = ""

class ConversationMemory:
    """Enhanced conversation memory with API optimization"""

    def __init__(self):
        """Initialize conversation memory"""
        self.slots: Dict[str, Any] = {}
        self.history: List[ConversationTurn] = []
        self.current_state: DialogueState = DialogueState.GREETING
        self.context_summary: str = ""

        # Initialize all slots as None
        for slot_name in REQUIRED_SLOTS:
            self.slots[slot_name] = None

    def update_slot(self, slot: str, value: Any) -> bool:
        """Update a conversation slot with new value"""
        if slot not in REQUIRED_SLOTS:
            logger.warning(f"Unknown slot: {slot}")
            return False

        old_value = self.slots.get(slot)
        self.slots[slot] = value

        logger.info(f"Updated slot '{slot}': {old_value} -> {value}")
        return True

    def update_slots_preserving_context(self, new_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Update slots while preserving existing context unless explicitly overridden"""

        # For slot_updation intent, preserve existing values
        for slot_name, new_value in new_slots.items():
            if slot_name != "user_intent" and new_value is not None:
                self.slots[slot_name] = new_value
                # Keep existing values for slots not mentioned

        logger.info(f"Updated slots preserving context: {self.slots}")
        return self.slots.copy()


    def get_all_slots(self) -> Dict[str, Any]:
        """Get ALL slots including null values"""
        return self.slots.copy()

    def display_all_slots(self) -> Dict[str, Any]:
        """Display all slots for debugging - includes nulls"""
        return {k: v if v is not None else "null" for k, v in self.slots.items()}

    def get_missing_slots(self) -> List[str]:
        """Get list of slots that are still empty"""
        missing = []
        for slot_name in REQUIRED_SLOTS:
            if not self.slots.get(slot_name):
                missing.append(slot_name)
        return missing

    def get_filled_slots(self) -> Dict[str, Any]:
        """Get all slots that have been filled with values"""
        return {k: v for k, v in self.slots.items() if v is not None}

    def add_turn(self, user_message: str, system_response: str,
                 intent: str, confidence: float, slots_updated: Dict = None):
        """Add a new conversation turn to history"""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            system_response=system_response,
            intent=intent,
            confidence=confidence,
            slots_updated=slots_updated or {}
        )
        self.history.append(turn)
        self._update_context_summary()

    def _update_context_summary(self):
        """Update internal context summary"""
        filled = self.get_filled_slots()
        if filled:
            slot_str = ", ".join([f"{k}={v}" for k, v in filled.items()])
            self.context_summary = f"Prefs: {slot_str}"
        else:
            self.context_summary = "No preferences set"

    def replace_all_slots(self, new_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Replace all slots with new values (for new_query intent)"""
        # Clear existing slots first
        for slot_name in REQUIRED_SLOTS:
            self.slots[slot_name] = None

        # Set new values
        for slot_name, value in new_slots.items():
            if slot_name in REQUIRED_SLOTS and value is not None:
                self.slots[slot_name] = value

        logger.info(f"Replaced all slots: {self.slots}")
        return self.slots.copy()

    def clear(self):
        """Reset conversation memory to initial state"""
        self.slots = {"dietary": None, "cuisine_1": None, "cuisine_2": None, "item_name": None, "price": None, "meal_type": None, "label": None}
        self.history = []
        self.current_state = DialogueState.GREETING
        self.context_summary = ""
        logger.info("Conversation memory cleared")

print("✅ ConversationMemory class defined")
print("🧠 Memory management with slot tracking enabled")



def history_to_json(memory_history: List['ConversationTurn']) -> List[dict]:
    return [
        {
            "timestamp": turn.timestamp,
            "user_message": turn.user_message,
            "system_response": turn.system_response,
            "intent": turn.intent,
            "confidence": turn.confidence,
            "slots_updated": turn.slots_updated,
            "action_state": turn.action_state,
        }
        for turn in memory_history
    ]