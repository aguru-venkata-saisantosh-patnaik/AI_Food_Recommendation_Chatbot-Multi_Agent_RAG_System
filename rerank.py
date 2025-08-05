# reranking of retrieved docs

import json
import openai
from typing import List, Dict, Any
from datetime import datetime
import logging

from rerank_prompts import part1, part2

logger = logging.getLogger(__name__)


class TwoStageContextualRerankerJSON:
    """
    Two-stage contextual reranker with JSON output parsing
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()

        # Store the JSON prompts
        self.stage1_prompt = part1
        self.stage2_prompt = part2

    def rerank_with_context(self, documents: List[Dict], conversation_history: List[Dict]) -> Dict:
        """
        Main two-stage reranking method with JSON parsing
        """
        try:
            # Stage 1: Context Analysis & Condition Generation
            stage1_result = self._stage1_context_analysis(documents, conversation_history)

            # Stage 2: Final Ranking & Reasoning Validation
            stage2_result = self._stage2_final_ranking(stage1_result)

            # Combine results
            final_result = {
                "stage1_analysis": stage1_result,
                "stage2_ranking": stage2_result,
                "final_combined_query": stage1_result.get("final_combined_query", ""),
                "temporal_context": stage1_result.get("temporal_context", ""),
                "user_journey": stage1_result.get("user_journey", ""),
                "ranking_conditions": stage1_result.get("ranking_conditions", []),
                "top_10_documents": stage2_result.get("top_10_documents", []),
                "quality_assurance": stage2_result.get("quality_assurance", {})
            }

            return final_result

        except Exception as e:
            logger.error(f"Two-stage contextual reranking failed: {e}")
            return self._fallback_reranking(documents, conversation_history)

    def _stage1_context_analysis(self, documents: List[Dict], conversation_history: List[Dict]) -> Dict:
        """
        Stage 1: Context Analysis & Condition Generation with JSON output
        """
        try:
            # Format inputs for Stage 1
            formatted_docs = self._format_documents_for_llm(documents)
            formatted_history = self._format_conversation_history(conversation_history)

            # Create Stage 1 prompt
            stage1_full_prompt = f"""
{self.stage1_prompt}

<< CONVERSATION HISTORY >>
{formatted_history}

<< AVAILABLE DOCUMENTS >>
{formatted_docs}

Analyze the conversation history and available documents. Return ONLY valid JSON as specified in the format above.
"""

            # Make Stage 1 API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": stage1_full_prompt}
                ],
                temperature=0.1,
                max_tokens=10000
            )

            response_content = response.choices[0].message.content.strip()

            print("=== STAGE 1: CONTEXT ANALYSIS ===")
            print(response_content)
            print("\n" + "="*50 + "\n")

            # Parse JSON response
            parsed_result = self._parse_json_response(response_content, "Stage 1")
            # parsed_result["raw_response"] = response_content

            input_doc_count = len(documents)
            evaluated_doc_count = len(parsed_result.get("document_evaluations", []))
            print(f'input docs {input_doc_count} and output docs {evaluated_doc_count}')


            return parsed_result

        except Exception as e:
            logger.error(f"Stage 1 analysis failed: {e}")
            raise e

    def _stage2_final_ranking(self, stage1_result: Dict) -> Dict:
        """
        Stage 2: Final Ranking & Reasoning Validation with JSON output
        """
        try:
            # Format Stage 1 results for Stage 2
            stage1_summary = json.dumps(stage1_result, indent=2, ensure_ascii=False)

            # Create Stage 2 prompt
            stage2_full_prompt = f"""
{self.stage2_prompt}

<< STAGE 1 ANALYSIS RESULTS >>
{stage1_summary}

Based on the contextual conditions and document evaluations from Stage 1, provide the final top 10 ranked recommendations. Return ONLY valid JSON as specified in the format above.
"""

            # Make Stage 2 API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": stage2_full_prompt}
                ],
                temperature=0.1,
                max_tokens=2500
            )

            response_content = response.choices[0].message.content.strip()

            print("=== STAGE 2: FINAL RANKING ===")
            print(response_content)
            print("\n" + "="*50 + "\n")

            # Parse JSON response
            parsed_result = self._parse_json_response(response_content, "Stage 2")
            parsed_result["raw_response"] = response_content

            return parsed_result

        except Exception as e:
            logger.error(f"Stage 2 ranking failed: {e}")
            raise e

    def _parse_json_response(self, response_content: str, stage: str) -> Dict:
        """
        Parse JSON response from LLM - much simpler than regex parsing
        """
        try:
            # Try to parse directly as JSON
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"{stage} JSON parsing failed: {e}")

            # Try to extract JSON from markdown code blocks
            try:
                import re
                json_match = re.search(r'``````', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # Try to find JSON object in response
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response_content[json_start:json_end]
                    return json.loads(json_str)

            except Exception as fallback_error:
                logger.error(f"{stage} fallback JSON parsing failed: {fallback_error}")

            # Return error structure
            return {
                "error": f"{stage} JSON parsing failed",
                "raw_content": response_content,
                "parse_error": str(e)
            }

    def _format_documents_for_llm(self, documents: List[Dict]) -> str:
        """Format documents for LLM processing"""
        formatted_docs = []

        for doc in documents:
            doc_info = {
                "id": doc.get('id', 'unknown'),
                "food_name": doc.get('metadata', {}).get('food', 'Unknown'),
                "restaurant": doc.get('metadata', {}).get('restaurant', 'Unknown'),
                "cuisine_1": doc.get('metadata', {}).get('cuisine_1', ''),
                "cuisine_2": doc.get('metadata', {}).get('cuisine_2', ''),
                "dietary": doc.get('metadata', {}).get('dietary', ''),
                "f_rating": doc.get('metadata', {}).get('f_rating', 0),
                "r_rating": doc.get('metadata', {}).get('r_rating', 0),
                "f_price": doc.get('metadata', {}).get('f_price', 0),
                "label": doc.get('metadata', {}).get('label', ''),
                "popularity": doc.get('metadata', {}).get('popularity', ''),
                "location": doc.get('metadata', {}).get('location', ''),
                "page_content": doc.get('page_content', '')
            }
            formatted_docs.append(doc_info)

        return json.dumps(formatted_docs, indent=2, ensure_ascii=False)

    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM analysis"""
        formatted_history = []

        for turn in conversation_history:
            timestamp = turn.get('timestamp', 'unknown')
            message = turn.get('user_message', '')


            # Parse timestamp if it's a string
            if isinstance(timestamp, str):
                try:
                    parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = parsed_time.strftime('%H:%M')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = str(timestamp)

            formatted_history.append(f"{formatted_time}: \"{message}\"")

        return "\n".join(formatted_history)

    def _fallback_reranking(self, documents: List[Dict], conversation_history: List[Dict]) -> Dict:
        """Fallback reranking when both stages fail"""

        # Simple fallback based on ratings and prices
        scored_docs = []

        for doc in documents:
            metadata = doc.get('metadata', {})
            f_rating = metadata.get('f_rating', 0)
            f_price = metadata.get('f_price', 1000)

            # Simple scoring: rating / (price/100)
            score = f_rating / max((f_price / 100), 1)

            scored_docs.append({
                "doc": doc,
                "score": score
            })

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Create fallback response in JSON format
        top_10_docs = []
        for i, scored_doc in enumerate(scored_docs[:10], 1):
            doc = scored_doc["doc"]
            metadata = doc.get('metadata', {})

            top_10_docs.append({
                "rank": i,
                "doc_id": doc.get('id', f'doc_{i}'),
                "food_name": metadata.get('food', 'Unknown'),
                "score": {
                    "critical": False,
                    "high": False,
                    "medium": False,
                    "low": False
                },
                "reasoning": f"Fallback ranking based on rating-price ratio ({scored_doc['score']:.2f}) due to API error."
            })

        return {
            "stage1_analysis": {"error": "Stage 1 failed"},
            "stage2_ranking": {"error": "Stage 2 failed"},
            "final_combined_query": "fallback query from conversation",
            "temporal_context": "fallback_context",
            "user_journey": "error_recovery",
            "ranking_conditions": [
                {
                    "priority": "HIGH",
                    "emoji": "🟡",
                    "description": "Fallback ranking using rating-price ratio",
                    "reasoning": "API failure recovery mode",
                    "measurable_criteria": "f_rating / (f_price/100)",
                    "document_field": "computed_score"
                }
            ],
            "top_10_documents": top_10_docs,
            "quality_assurance": {"status": "fallback_mode"}
        }




# To use this code:
# 1. Store your prompts as part1 and part2 variables (JSON versions above)
# 2. Initialize: reranker = TwoStageContextualRerankerJSON()
# 3. Call: result = reranker.rerank_with_context(all_docs, history_json_list)
