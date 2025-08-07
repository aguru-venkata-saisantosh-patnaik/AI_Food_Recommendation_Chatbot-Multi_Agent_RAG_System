# ────────────────────────────────────────────────────────────
# 📍 Cell 3  — Core Enums and Configuration  (MODIFIED - FIXED)
# ────────────────────────────────────────────────────────────
from enum import Enum
import time
import openai
from openai import OpenAI
import logging

class DialogueState(Enum):
    GREETING          = "greeting"
    INTENT_DETECTION  = "intent_detection"
    SLOT_FILLING      = "slot_filling"
    RECOMMEND         = "recommend"
    FEEDBACK          = "feedback"
    CLOSING           = "closing"

class IntentType(Enum):
    RECOMMEND        = "recommend"
    FILTER_UPDATE    = "filter_update"
    CLARIFICATION    = "clarification"
    FEEDBACK         = "feedback"
    GREETING         = "greeting"
    GOODBYE          = "goodbye"
    OTHER            = "other"

# ------------------------------------------------------------------
# REQUIRED_SLOTS - updated with cuisine_1, cuisine_2, item_name
# ------------------------------------------------------------------
REQUIRED_SLOTS = {
    "dietary": {
        "type": "string",
        "values": ["veg", "nonveg", "vegan"],
        "description": "User's dietary preference",
        "required": False,
    },
    "cuisine_1": {
        "type": "string",
        "values": [],                     # validated with _CUISINES list in Slot-Extraction cell
        "description": "Primary cuisine preference",
        "required": False,
    },
    "cuisine_2": {
        "type": "string",
        "values": [],
        "description": "Secondary cuisine preference (optional)",
        "required": False,
    },
    "item_name": {
        "type": "string",
        "values": [],                     # free text – no fixed choices
        "description": "Specific dish mentioned by user",
        "required": False,
    },
    "price": {
        "type": "integer",
        "min": 50,
        "max": 2000,
        "description": "Price in rupees or budget tier",
        "required": False,
    },
    "meal_type": {
        "type": "string",
        "values": ["breakfast", "lunch", "dinner", "snacks"],
        "description": "Type of meal",
        "required": False,
    },
    "label": {
        "type": "string",
        "values": [
            "bestseller", "must try", "chef's special", "new", "seasonal",
            "spicy", "dairy free", "vegan", "gluten free", "eggless available",
            "fodmap friendly", "not eligible for coupons", "not on pro"
        ],
        "description": "Menu label preference",
        "required": False,
    },
}

api_key = 'ENTER_OWN_KEY'

client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4o-mini"

def call_openai(prompt: str, temperature: float = 0.1) -> str:
    """
    Thin wrapper around the OpenAI ≥1.0 chat endpoint.
    Returns the assistant’s raw text reply.
    """
    try:
        response = client.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature = temperature,
            max_tokens  = 800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.requests.append(now)

rate_limiter = RateLimiter()
