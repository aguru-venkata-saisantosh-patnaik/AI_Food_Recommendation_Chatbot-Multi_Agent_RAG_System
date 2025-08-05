# Cell 5: Slot Extraction - Keep slots empty instead of "any" (FIXED)
import re
import json
import logging
from difflib import get_close_matches
from typing import Dict, Any, Optional

from utils import call_openai, rate_limiter, REQUIRED_SLOTS

logger = logging.getLogger(__name__)


# ----------------------------------------
# Helper dictionaries / constants
# ----------------------------------------
_MEAL_KEYWORDS = {
    "breakfast": ["breakfast", "brekkie", "morning"],
    "lunch":     ["lunch", "lunchtime", "noon"],
    "dinner":    ["dinner", "supper", "evening"],
    "snacks":    ["snack", "snacks", "tiffin", "munchies"],
}

# (same cuisine list you already had)
_CUISINES = [
    "pizzas","bakery","indian","fast food","chaat","beverages","desserts",
    "chinese","north indian","tandoor","american","thalis","snacks",
    "south indian","italian","street food","kebabs","biryani","salads",
    "pastas","continental","bengali","burgers","ice cream","tibetan","thai",
    "hyderabadi","sweets","lebanese","nepalese","mughlai","lucknowi",
    "healthy food","afghani","asian","combo","seafood","waffle",
    "italian-american","punjabi","arabian","barbecue","mexican",
    "ice cream cakes","gujarati","juices","jain","pan-asian","rajasthani",
    "mediterranean","burmese","oriental","maharashtrian","kerala","home food",
    "indonesian","middle eastern","grill","japanese","paan","greek",
    "chettinad","coastal","andhra","turkish","african","tex-mex",
    "oriya","british","mangalorean","bihari","keto","european","malaysian",
    "north eastern","sushi","french","korean","portuguese","naga","assamese",
    "steakhouse"
]


# ----------------------------------------
# Fuzzy meal-type helper
# ----------------------------------------
def _nearest_meal(word: str) -> Optional[str]:
    corpus = [w for syns in _MEAL_KEYWORDS.values() for w in syns]
    hit = get_close_matches(word.lower(), corpus, n=1, cutoff=0.75)
    if hit:
        for meal, syns in _MEAL_KEYWORDS.items():
            if hit[0] in syns:
                return meal
    return None

# ----------------------------------------
# FIXED: Simplified rule-based fallback extractor
# ----------------------------------------
# Cell 5: Fixed Slot Extraction with Better "No Restrictions" Handling
import re, json, logging
from difflib import get_close_matches
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def robust_fallback_slot_extraction(msg: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    txt = msg.lower()

    # 1️⃣ DIETARY - handle "no restrictions" properly
    if re.search(r"no restrictions?|no dietary|eat everything|not picky", txt):
        out["dietary"] = "nonveg"  # Default to nonveg for "no restrictions"
    elif re.search(r"\bnon[- ]?veg", txt):
        out["dietary"] = "nonveg"
    elif "vegan" in txt:
        out["dietary"] = "vegan"
    elif re.search(r"\bveg( |$)", txt):
        out["dietary"] = "veg"

    # 2️⃣ CUISINE handling remains the same...
    cuisines_found = []
    for c in _CUISINES:
        if c in txt:
            cuisines_found.append(c)
        if len(cuisines_found) >= 2:
            break

    if cuisines_found:
        out["cuisine_1"] = cuisines_found[0]
        if len(cuisines_found) > 1:
            out["cuisine_2"] = cuisines_found[1]

    # 3️⃣ ITEM_NAME - better flavor handling
    if "flavor" in txt or "flavour" in txt:
        # Extract flavor mentions and combine with existing item
        flavor_match = re.search(r"(\w+)\s+flavo?ur?", txt)
        if flavor_match:
            flavor = flavor_match.group(1)
            # This will be handled by the enhanced update logic
            out["item_clarification"] = flavor

    # 4️⃣ PRICE - handle budget discussions better
    if any(phrase in txt for phrase in ["budget", "spend", "afford", "cheap", "expensive"]):
        # If they mention budget but no specific amount, we'll ask for it
        out["price_mentioned"] = True

    price_patterns = [
        r"(?:under|below|less than|max(?:imum)?|upto|up to)\s*₹?\s*(\d{2,4})",
        r"(?:within|budget of|spend)\s*₹?\s*(\d{2,4})",
        r"₹\s*(\d{2,4})",
        r"(\d{2,4})\s*(?:rs|rupees|bucks|₹)",
    ]
    for pat in price_patterns:
        m = re.search(pat, txt)
        if m:
            val = int(m.group(1))
            if 50 <= val <= 5000:
                out["price"] = val
                break

    return out


# ----------------------------------------
# FIXED: Enhanced OpenAI-assisted extraction
# ----------------------------------------
def extract_slots_from_message(user_message: str, context: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]:
    try:
        # --- OpenAI call with explicit instructions to avoid "any" ---
        rate_limiter.wait_if_needed()
        ctx = context.get("filled_slots") if context else {}

        # FIXED: Enhanced prompt that properly extracts item_name
        enhanced_prompt = f"""
Extract food preferences from this message and determine user intent based on previous context.

SLOT & INTENT EXTRACTION RULES

GENERAL PRINCIPLES:
Only extract values IN THE USER'S CURRENT MESSAGE; never infer, assume, or generalize.
Never use "any," "all," "whatever," etc. for any slot. If user expressly declines to specify ("no preference," "anything is fine"), extract null for that slot.
For all slot updates, update ONLY what is stated, ignore or null the rest.
Always output a single valid JSON.

SLOT HANDLING:
dietary: Extract if user explicitly mentions veg, nonveg, vegan. Otherwise, null.
cuisine_1: Extract only if user explicitly specifies a cuisine matching the allowed list. Otherwise, null.
cuisine_2: Extract only if a second, distinct cuisine is explicitly mentioned.
item_name: Only if a specific food or dish name is named by the user.
price: Extract numerical value if the user states a budget/price.
meal_type: Extract if user specifically mentions breakfast/lunch/dinner/snacks.
label: Extract if the user offers a specific label (e.g., spicy, bestseller).
Null all other slots not directly addressed in the user's utterance for this turn.

INTENT DETECTION:
slot_updation: Use ONLY if the user's message adds/subtracts detail but does NOT imply a complete change (same dish/cuisine family, or explicit elaboration).
Null slots not changed by user message.
new_query: Trigger if user asks for a different food, especially with language like "now I want," "instead," "change to," or via a clear course switch (main to dessert, main to drink etc.), or if cuisine switches from current to another category.
Clear all previous values except those directly called for in the new message and fill slots based on current message only.

EDGE CASE RULES:
If user names a food item/dish or cuisine that does not match present context, trigger new_query.
Course switches (e.g., from main to dessert/drink) or explicit "now/instead/change to" language must trigger new_query.
If user gives a partial refinement (e.g., "under 400"), only fill the price slot, null for others.

Current preferences: {ctx}
User message: "{user_message}"


_CUISINES = [
    "pizzas","bakery","indian","fast food","chaat","beverages","desserts",
    "chinese","north indian","tandoor","american","thalis","snacks",
    "south indian","italian","street food","kebabs","biryani","salads",
    "pastas","continental","bengali","burgers","ice cream","tibetan","thai",
    "hyderabadi","sweets","lebanese","nepalese","mughlai","lucknowi",
    "healthy food","afghani","asian","combo","seafood","waffle",
    "italian-american","punjabi","arabian","barbecue","mexican",
    "ice cream cakes","gujarati","juices","jain","pan-asian","rajasthani",
    "mediterranean","burmese","oriental","maharashtrian","kerala","home food",
    "indonesian","middle eastern","grill","japanese","paan","greek",
    "chettinad","coastal","andhra","turkish","african","tex-mex",
    "oriya","british","mangalorean","bihari","keto","european","malaysian",
    "north eastern","sushi","french","korean","portuguese","naga","assamese",
    "steakhouse"
]

Extract these slots:
- dietary: "veg", "nonveg", "vegan", or null
- cuisine_1: specific cuisine type from _CUISINES mentioned or closely relatable by user, or null
- cuisine_2: second cuisine type from _CUISINES mentioned that differs from cuisine_1 if required or said by user for multi cuisine type food , or null
- item_name: specific food dish mentioned (biryani, tikka biryani, strawberry icecream, etc.), or null
- price: numerical price or budget, or null
- meal_type: "breakfast", "lunch", "dinner", "snacks", or null
- label: "bestseller", "spicy", "sweet", "dairy free", etc., or null

RESPOND IN THIS EXACT JSON FORMAT:
{{
 "user_intent": "slot_updation" or "new_query",
 "dietary": "specific value or null",
 "cuisine_1": "specific value or null",
 "cuisine_2": "specific value or null",
 "item_name": "specific value or null",
 "price": number or null,
 "meal_type": "specific value or null",
 "label": "specific value or null"
}}

EXAMPLES FOR CONTEXT PRESERVATION:

**Example 1 - SLOT_UPDATION with Context Preservation:**
Previous slots: {{"dietary": "vegan", "cuisine_1": "ice cream", "item_name": "ice cream", "price": null}}
User message: "under 400"
Response:
{{
 "user_intent": "slot_updation",
 "dietary": null,        // null = don't change existing value (preserve "vegan")
 "cuisine_1": null,      // null = don't change existing value (preserve "ice cream")
 "cuisine_2": null,      // null = no change
 "item_name": null,      // null = don't change existing value (preserve "ice cream")
 "price": 400,           // Update this slot only
 "meal_type": null,      // null = no change
 "label": null           // null = no change
}}

**Example 2 - SLOT_UPDATION with Flavor Addition:**
Previous slots: {{"dietary": "veg", "cuisine_1": "ice cream", "item_name": "ice cream", "price": 200}}
User message: "want peach flavoured"
Response:
{{
 "user_intent": "slot_updation",
 "dietary": null,            // preserve existing "veg"
 "cuisine_1": null,          // preserve existing "ice cream"
 "cuisine_2": null,          // no change
 "item_name": "peach ice cream", // combine with existing context
 "price": null,              // preserve existing 200
 "meal_type": null,          // no change
 "label": null               // no change
}}

**Example 3 - NEW_QUERY Intent (Clear Everything):**
Previous slots: {{"dietary": "veg", "cuisine_1": "ice cream", "item_name": "peach ice cream", "price": 400}}
User message: "actually, now I want some biryani"
Response:
{{
 "user_intent": "new_query",
 "dietary": null,         // clear previous context
 "cuisine_1": "biryani",  // new cuisine mentioned beacuase previous cuisine_1 was ice cream and it does not match with some biriyani type cuisine
 "cuisine_2": null,       // not mentioned
 "item_name": "biryani",  // new item mentioned
 "price": null,           // clear previous price
 "meal_type": null,       // not mentioned
 "label": null            // not mentioned
}}

**Example 4 - SLOT_UPDATION with Multiple Updates:**
Previous slots: {{"dietary": "nonveg", "cuisine_1": "burgers", "item_name": "burger", "price": null}}
User message: "make it italian style under 300"
Response:
{{
 "user_intent": "slot_updation",
 "dietary": null,                    // preserve existing "nonveg"
 "cuisine_1": null,                  // since cuisine_1 is already filled and another type is mentioned, keep it as null
 "cuisine_2": "italian",             // update secondary cuisine to Italian
 "item_name": null,                  // preserve existing "burger"
 "price": 300,                       // new price mentioned
 "meal_type": null,                  // no change
 "label": null                       // no change
}}

**Example 5 - SLOT_UPDATION with Cuisine Addition:**
Previous slots: {{"dietary": "veg", "cuisine_1": "burgers", "item_name": "burger", "price": 300}}
User message: "also ensure it italian"
Response:
{{
 "user_intent": "slot_updation",
 "dietary": null,           // preserve existing "veg"
 "cuisine_1": null,       // since cuisine_1 is already filled and another type is mentioned, keep it as null
 "cuisine_2": "italian",    // update secondary cuisine to Italian
 "item_name": null,         // preserve existing "burger"
 "price": null,             // preserve existing 300
 "meal_type": null,         // no change
 "label": null              // no change
}}

**Example 6 - SLOT_UPDATION with Cuisine Addition:**
Previous slots: {{"dietary": "veg", "cuisine_1": "burgers", "item_name": "burger", "price": 300}}
User message: "not italian maybe mexiacn"
Response:
{{
 "user_intent": "slot_updation",
 "dietary": null,           // preserve existing "veg"
 "cuisine_1": null,       // since cuisine_1 is already filled and another type is mentioned, keep it as null
 "cuisine_2": "mexican",    // update secondary cuisine to mexican
 "item_name": null,         // preserve existing "burger"
 "price": null,             // preserve existing 300
 "meal_type": null,         // no change
 "label": null              // no change
}}

"""




        raw = call_openai(enhanced_prompt).strip()
        print(ctx)
        print(raw)
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        parsed = json.loads(raw)

    except Exception as e:
        logger.warning(f"OpenAI extraction failed → {e}; using fallback.")
        parsed = robust_fallback_slot_extraction(user_message)

    return {k: v for k, v in parsed.items() if v not in (None, "", "null", "any")}



# Convenience wrapper used elsewhere
def extract_slots(user_message: str, memory=None):
    ctx = {"filled_slots": memory.get_filled_slots()} if memory else None
    return extract_slots_from_message(user_message, ctx)

print("✅ Slot extraction updated - no 'any' values, keeps slots empty for specificity")
print("🔍 System will continue asking until specific preferences are provided")