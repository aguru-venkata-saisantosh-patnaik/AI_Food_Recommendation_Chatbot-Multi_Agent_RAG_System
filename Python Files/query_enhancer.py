# Query Enhancement and Filter Building

import re
import json
import logging
from typing import Dict, Any, Optional, Union, List

from utils import client, rate_limiter
from slot_extract import extract_slots_from_message

logger = logging.getLogger(__name__)


class OpenAIQueryEnhancer:
    """Enhanced query builder and slot extractor using OpenAI"""

    def __init__(self):
        # Cuisine mapping for filter building
        self.cuisine_mapping = {
            "beverages": ["beverages", "juices"],
            "chinese": ["chinese", "asian", "pan-asian", "oriental"],
            "indian": ["indian", "north indian", "south indian", "bengali", "punjabi", "mughlai", "lucknowi", "gujarati", "rajasthani", "maharashtrian", "kerala", "andhra", "chettinad", "hyderabadi", "bihari", "oriya", "mangalorean", "north eastern", "naga", "assamese", "jain"],
            "thai": ["thai"],
            "italian": ["italian", "italian-american"],
            "mexican": ["mexican", "tex-mex"],
            "japanese": ["japanese", "sushi"],
            "korean": ["korean"],
            "french": ["french"],
            "american": ["american"],
            "continental": ["continental", "european"],
            "fast food": ["fast food"],
            "healthy food": ["healthy food", "keto"],
            "desserts": ["desserts", "sweets", "ice cream", "ice cream cakes"],
            "bakery": ["bakery", "waffle"],
            "snacks": ["snacks", "chaat", "paan"],
            "biryani": ["biryani", "biryani - shivaji military hotel"],
            "pizzas": ["pizzas"],
            "pastas": ["pastas"],
            "burgers": ["burgers"],
            "salads": ["salads"],
            "seafood": ["seafood", "coastal"],
            "tandoor": ["tandoor", "kebabs", "grill"],
            "street food": ["street food", "svanidhi street food vendor"],
            "thalis": ["thalis"],
            "combo": ["combo"],
            "middle eastern": ["middle eastern", "arabian", "lebanese", "turkish", "afghani"],
            "nepalese": ["nepalese"],
            "tibetan": ["tibetan"],
            "mediterranean": ["mediterranean", "greek"],
            "burmese": ["burmese"],
            "indonesian": ["indonesian"],
            "malaysian": ["malaysian"],
            "barbecue": ["barbecue", "steakhouse"],
            "british": ["british"],
            "portuguese": ["portuguese"],
            "african": ["african"],
            "home food": ["home food"],
            "cafe": ["cafe"],
        }
        self.price_tier_mapping = {
            "budget": {"min": 50, "max": 200},
            "affordable": {"min": 200, "max": 500},
            "premium": {"min": 500, "max": 1000},
            "luxury": {"min": 1000, "max": 2000}
        }
        self.model = "gpt-4o-mini"
        self.temperature = 0.2


    def build_enhanced_query(self, memory: 'ConversationMemory', intent: str,) -> Dict[str, Any]:
        """Build enhanced query structure from conversation state with LLM refinement"""
        filled_slots = memory.get_filled_slots()

        # Step 1: Construct base semantic query and filter
        base_query = self._construct_semantic_query(filled_slots, intent)
        base_filter = self._build_chroma_filter(filled_slots)

        # Step 2: Use LLM to refine query and filter based on user input
        if len(memory.history):
            try:
                refined_result = self._refine_with_llm(
                    base_query=base_query,
                    base_filter=base_filter,
                    filled_slots=filled_slots,
                    memory=memory
                )
                print("llm call for refinement")
                return refined_result
            except Exception as e:
                logger.error(f"LLM refinement failed: {e}, using base query/filter")

        # Fallback to original implementation
        return {
            "timestamp": "dxfcgvhbjn",
            "query": base_query,
            "filter": base_filter,
            "clarifying_questions": self._generate_clarifying_questions(
                memory.get_missing_slots(), filled_slots
            )
        }

    def _refine_with_llm(self, base_query: str, base_filter: Any, filled_slots: Dict, memory: 'ConversationMemory') -> Dict[str, Any]:
        """Use LLM to refine query and filter based on user input"""

        # Build refinement prompt
        prompt = self._build_query_refinement_prompt(
            base_query, base_filter, filled_slots, memory
        )

        # Call OpenAI API
        rate_limiter.wait_if_needed()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert search query and filter refinement assistant specializing in food delivery recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        response_text = response.choices[0].message.content.strip()
        print(f"LLM Refinement Response: {response_text}")

        # Parse JSON response
        return self._parse_refinement_response(response_text, base_query, base_filter)

    def _build_query_refinement_prompt(self, base_query: str, base_filter: Any,
                                     filled_slots: Dict, memory: 'ConversationMemory') -> str:
        """Build prompt for LLM query refinement"""

        # Format conversation history in line-wise format
        history_lines = []
        for turn in memory.history:
            history_lines.append(f"User: {turn.user_message}")
            history_lines.append(f"System: {turn.system_response}")
            history_lines.append(f"Slots Updated: {turn.slots_updated}")
            history_lines.append("---")

        history_text = "\n".join(history_lines) if history_lines else "No previous conversation"

        prompt = f'''
You are a food search query refinement expert that structures user preferences into optimized search queries.

Your goal is to structure the user's overall food preferences from conversation history into the request schema provided below.

<< Structured Request Schema >>

When responding use a markdown code snippet with a JSON object formatted in the following schema:

{{
"query": "string", // text string to compare with document contents
"filter": "object | NO_FILTER" // a Chroma-compatible JSON filter or "NO_FILTER"
}}

text

The query should only include keywords relevant to matching the content of documents. Filter conditions should only be included in the "filter" field and not repeated in the query.

<< CONVERSATION CONTEXT >>

**Conversation History:**
{history_text}

**Current Filled Slots:**
{json.dumps(filled_slots, indent=2)}

**Base Query Generated:** "{base_query}"
**Base Filter Generated:** {json.dumps(base_filter, indent=2)}

<< REFINEMENT TASK >>

Analyze the ENTIRE conversation to understand user's final food preferences and create an optimized query and filter.

<< CRITICAL PRICE HANDLING RULES >>

1. **Strict conditions** ("under 300", "below 250", "maximum 400", "strictly under"):
   - Use exact upper limit: {{"f_price": {{"$lte": AMOUNT}}}}

2. **Approximate conditions** ("about 300", "around 250", "roughly 400"):
   - Use 15% variance: {{"$and": [{{"f_price": {{"$gte": AMOUNT*0.85}}}}, {{"f_price": {{"$lte": AMOUNT*1.15}}}}]}}

3. **Range conditions** ("between 200 and 400"):
   - Use exact range: {{"$and": [{{"f_price": {{"$gte": 200}}}}, {{"f_price": {{"$lte": 400}}}}]}}

<< QUERY REFINEMENT RULES >>

- Remove duplicate terms (e.g., "biryani biryani" → "biryani")
- Combine related food terms intelligently (e.g., "dum biriyani" should be "dum biryani")
- Include flavor/style descriptors mentioned by user
- Keep cuisine and item_name terms prominent
- Focus on the user's FINAL food preference from conversation

<< FILTER ENHANCEMENT RULES >>

- Use ChromaDB compatible syntax with proper operators
- Handle multiple conditions with $and/$or as needed
- Use only attribute names that exist in the data source
- If no filters needed, return "NO_FILTER"

<< DATA SOURCE ATTRIBUTES >>

Available filter attributes:
- **dietary**: "veg", "nonveg", "vegan"
- **cuisine_1**, **cuisine_2**: cuisine types from your mapping
- **f_price**: integer price in INR
- **f_rating**: float rating 0-5
- **location**: city names
- **label**: "bestseller", "must try", "spicy", etc.

<< EXAMPLES >>

**Example 1 - Conversation Leading to Biryani:**
History shows: User wants biryani → specifies nonveg → mentions under 400 → updates to dum biryani

{{
  "query": "dum biryani",
  "filter": {{
    "$and": [
      {{"dietary": {{"$eq": "nonveg"}}}},
      {{"f_price": {{"$lte": 400}}}},
      {{"$or": [
        {{"cuisine_1": {{"$eq": "biryani"}}}},
        {{"cuisine_2": {{"$eq": "biryani"}}}}
      ]}}
    ]
  }}
}}

text

**Example 2 - Ice Cream with Flavor:**
History shows: User wants ice cream → specifies vegan → mentions under 400 → adds strawberry flavor

{{
  "query": "strawberry ice cream",
  "filter": {{
    "$and": [
      {{"dietary": {{"$eq": "vegan"}}}},
      {{"f_price": {{"$lte": 400}}}},
      {{"$or": [
        {{"cuisine_1": {{"$eq": "ice cream"}}}},
        {{"cuisine_2": {{"$eq": "ice cream"}}}}
      ]}}
    ]
  }}
}}

text

**Example 3 - Approximate Price:**
User mentions "about 300 rupees for pizza"

{{
  "query": "pizza",
  "filter": {{
    "$and": [
      {{"$and": [
        {{"f_price": {{"$gte": 255}}}},
        {{"f_price": {{"$lte": 345}}}}
        ]}},
      {{"$or": [
        {{"cuisine_1": {{"$eq": "pizzas"}}}},
        {{"cuisine_2": {{"$eq": "pizzas"}}}}
      ]}}
    ]
  }}
}}

text

**Example 4 - No Filter Needed:**
User just wants "general food recommendations"

{{
  "query": "food recommendations",
  "filter": "NO_FILTER"
}}

text

<< CRITICAL INSTRUCTIONS >>

1. Analyze the COMPLETE conversation flow to understand final user intent
2. Create ONE optimized query representing their final food choice
3. Build ONE comprehensive filter covering all their requirements
4. Remove query duplications and combine intelligently
5. Use proper ChromaDB filter syntax with correct operators
6. Return ONLY the JSON structure requested

Focus on the user's FINAL preferences after all conversation turns!
'''
        return prompt

    def _parse_refinement_response(self, response_text: str, fallback_query: str,
                             fallback_filter: Any) -> Dict[str, Any]:
        """Parse LLM refinement response matching query_construct_prompt format"""

        try:
            # Clean the response text
            response_text = response_text.strip()

            if '```' in response_text:
                json_str_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if not json_str_match:
                    raise ValueError("No JSON code block found in LLM response")
                json_str = json_str_match.group(1)
            else:
                json_str_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_str_match:
                    raise ValueError("No JSON object found in LLM response")
                json_str = json_str_match.group(0)

            parsed_response = json.loads(json_str)

            # Validate and extract required fields
            refined_query = parsed_response.get("query", fallback_query)
            refined_filter = parsed_response.get("filter", fallback_filter)

            # Handle string "NO_FILTER" case
            if isinstance(refined_filter, str) and refined_filter == "NO_FILTER":
                refined_filter = "NO_FILTER"

            print(f"✅ Successfully refined query: '{refined_query}'")
            print(f"✅ Successfully refined filter: {refined_filter}")

            return {
                "query": refined_query,
                "filter": refined_filter,
                "clarifying_questions": []  
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM refinement response: {e}")
            logger.error(f"Raw response was: {response_text}")
            print(f"❌ Using fallback - Original query: '{fallback_query}'")

            return {
                "query": fallback_query,
                "filter": fallback_filter,
                "clarifying_questions": []
            }


    def extract_slots_from_message(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract slots using OpenAI (compatibility method)"""
        return extract_slots_from_message(user_message, context)

    def _construct_semantic_query(self, filled_slots: Dict, intent: str) -> str:
        """FIXED: Construct semantic search query using cuisine_1, cuisine_2, item_name, and label"""
        query_parts = [] 

        # Primary & secondary cuisine
        if filled_slots.get("cuisine_1"):
            query_parts.append(filled_slots["cuisine_1"]) 
        if filled_slots.get("cuisine_2") and filled_slots["cuisine_2"] != filled_slots.get("cuisine_1"):
            query_parts.append(filled_slots["cuisine_2"]) 

        # Specific dish name
        if filled_slots.get("item_name"):
            query_parts.append(filled_slots["item_name"]) 

        # Label preferences (as requested - included in semantic search)
        if filled_slots.get("label"):
            query_parts.append(filled_slots["label"])

        # Add meal context (optional, can be included or removed based on preference)
        if filled_slots.get("meal_type"):
            query_parts.append(filled_slots["meal_type"])

        # Default fallback
        if not query_parts:
            query_parts.append("food")

        return " ".join(query_parts)

    def _build_chroma_filter(self, filled_slots: Dict) -> Union[Dict, str]:
        """FIXED: Build ChromaDB-compatible filter using cuisine_1/cuisine_2"""
        filter_conditions = []

        # Dietary filter
        if filled_slots.get("dietary"):
            dietary_value = filled_slots["dietary"]
            filter_conditions.append({"dietary": {"$eq": dietary_value}})

        # Cuisine filter - handle both cuisine_1 and cuisine_2
        cuisine_variants = []
        if filled_slots.get("cuisine_1"):
            cuisine1 = filled_slots["cuisine_1"].lower()
            cuisine_variants.extend(self.cuisine_mapping.get(cuisine1, [cuisine1]))
        if filled_slots.get("cuisine_2"):
            cuisine2 = filled_slots["cuisine_2"].lower()
            cuisine_variants.extend(self.cuisine_mapping.get(cuisine2, [cuisine2]))

        if cuisine_variants:
            cuisine_or_conditions = []
            for variant in cuisine_variants:
                cuisine_or_conditions.extend([
                    {"cuisine_1": {"$eq": variant}},
                    {"cuisine_2": {"$eq": variant}}
                ])
            if cuisine_or_conditions:
                filter_conditions.append({"$or": cuisine_or_conditions})


        # Combine conditions
        if len(filter_conditions) == 0:
            return "NO_FILTER"
        elif len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            return {"$and": filter_conditions}

    def _generate_clarifying_questions(self, missing_slots: List[str], filled_slots: Dict) -> List[str]:
        """Generate clarifying questions for missing slots"""
        questions = []

        question_templates = {
            "dietary": "Veg, nonveg, or vegan?",
            "cuisine_1": "Any particular cuisine you prefer?",
            "cuisine_2": "Any second cuisine preference?",
            "item_name": "Any specific dish you're craving?",
            "price": "What's your budget or price range?",
            "meal_type": "Is this for breakfast, lunch, dinner, or snacks?",
            "label": "Any specific preferences (spicy, sweet, bestseller)?"
        }

        for slot in missing_slots[:2]: 
            if slot in question_templates:
                questions.append(question_templates[slot])

        return questions


