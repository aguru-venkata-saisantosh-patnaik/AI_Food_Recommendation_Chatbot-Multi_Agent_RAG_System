# orchestrator

import yaml
import logging
from typing import List, Dict 
from datetime import datetime

from memory import ConversationMemory, history_to_json
from conversation_agent import OpenAIConversationAgent
from query_enhancer import OpenAIQueryEnhancer
from shards_retrieval import retrieve_all_docs_with_llm_query
from rerank import TwoStageContextualRerankerJSON
from utils import api_key
from embeddings import setup_embeddings_cpu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class RecommenderOrchestrator:
    def __init__(self, config_path: str):
        # Load configuration from YAML
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Shared resources
        self.shard_info_path = self.config["shard_info_path"]
        self.memory = ConversationMemory()
        self.embeddings = setup_embeddings_cpu()
        self.conv_agent = OpenAIConversationAgent(
            memory=self.memory,
            embeddings=self.embeddings
        )
        self.query_enhancer = OpenAIQueryEnhancer()

        self.reranker_agent = TwoStageContextualRerankerJSON(
            model=self.config["rerank_model"],
            api_key=api_key
        )

    # Remove the initialize_user method entirely or make it empty
    def initialize_user(self):
        """
        No longer needed - clustering functionality removed
        """
        logger.info("Session initialized without user clustering")
        pass

    def handle_chat(self, user_message: str, recommendations_context: dict = None) -> dict:
        """
        Handle a single chat turn: update slots/intent, optionally run retrieval & rerank,
        then return the assembled response dict.
        """
        # 1. Delegate to conversation agent
        conv_response = self.conv_agent.handle_turn(user_message, recommendations_context)

        # 2. If ready, run retrieval + rerank pipeline
        if conv_response["action"] in ("SEARCH", "SEARCH_READY"):
            enhanced_query = self.query_enhancer.build_enhanced_query(self.conv_agent.memory, "recommendation")
            all_docs = retrieve_all_docs_with_llm_query(enhanced_query["query"], enhanced_query["filter"], self.shard_info_path, self.embeddings)
            history_json = history_to_json(self.conv_agent.memory.history)
            rerank_res = self.reranker_agent.rerank_with_context(all_docs, history_json, enhanced_query)


            enriched_top_docs = self._enrich_top_docs_with_metadata(
                rerank_res["top_10_documents"], 
                all_docs
            )

            conv_response["recommendations"] = enriched_top_docs

            conv_response["ranking_conditions"] = rerank_res["ranking_conditions"]

            print("reranking is done",'\n\n')
            print(rerank_res["top_10_documents"], '\n\n')
            print("ranking conditions:", rerank_res["ranking_conditions"])


            # Track search but don't end conversation
            self.conv_agent.search_history.append({
                "query": enhanced_query["query"],
                "filter": enhanced_query["filter"],
                "timestamp": datetime.now().isoformat(),
                "conditions": rerank_res["ranking_conditions"],
                "top_docs": rerank_res["top_10_documents"]
            })


        return conv_response

    def _enrich_top_docs_with_metadata(self, top_10_documents: List[Dict], all_docs: List[Dict]) -> List[Dict]:
        """
        Enrich top 10 documents with full metadata from original retrieved documents
        """
        # Create lookup dictionary for fast access
        docs_lookup = {doc.get('id', ''): doc for doc in all_docs}

        enriched_docs = []

        for ranked_doc in top_10_documents:
            doc_id = ranked_doc.get('doc_id', '')

            original_doc = docs_lookup.get(doc_id)

            if original_doc:
                # Create enriched document with ranking info + full metadata
                enriched_doc = {
                    **ranked_doc,  
                    "metadata": original_doc.get('metadata', {}),
                }
            else:
                # Fallback if document not found
                enriched_doc = ranked_doc
                enriched_doc["metadata"] = {}
                enriched_doc["page_content"] = ""
            
            enriched_docs.append(enriched_doc)
        
        return enriched_docs



    # displaying progress bar in ui
    def handle_chat_with_progress_steps(self, user_message: str, recommendations_context: dict = None, progress_callback=None):
        """
    Modified handle_chat that calls progress_callback at each step
    """
    
        conv_response = self.conv_agent.handle_turn(user_message, recommendations_context)
    
        if conv_response["action"] in ("SEARCH", "SEARCH_READY"):
        
        # Step 1: Query Enhancement
            if progress_callback:
                progress_callback(0.1, "🔍 Refining query to search across shards...")
            enhanced_query = self.query_enhancer.build_enhanced_query(self.conv_agent.memory, "recommendation")
        
        # Step 2: Document Retrieval  
            if progress_callback:
                progress_callback(0.4, "📚 Searching across database shards...")
            all_docs = retrieve_all_docs_with_llm_query(
                enhanced_query["query"], enhanced_query["filter"], 
                self.shard_info_path, self.embeddings
            )
        
        # Step 3: Reranking
            if progress_callback:
                progress_callback(0.7, "🎯 Evaluating and reranking recommendations...")
            history_json = history_to_json(self.conv_agent.memory.history)
            rerank_res = self.reranker_agent.rerank_with_context(all_docs, history_json, enhanced_query)
        
        # Step 4: Metadata Enrichment
            if progress_callback:
                progress_callback(0.9, "✨ Finalizing recommendations...")
            enriched_top_docs = self._enrich_top_docs_with_metadata(
                rerank_res["top_10_documents"], all_docs
            )
        
            conv_response["recommendations"] = enriched_top_docs
            conv_response["ranking_conditions"] = rerank_res["ranking_conditions"]
        
        # Final step
            if progress_callback:
                progress_callback(1.0, "✅ Complete!")
        
        # Track search history
            self.conv_agent.search_history.append({
                "query": enhanced_query["query"],
                "filter": enhanced_query["filter"],
                "timestamp": datetime.now().isoformat(),
                "conditions": rerank_res["ranking_conditions"],
                "top_docs": enriched_top_docs
            })
    
        return conv_response

    def get_search_history(self) -> List[Dict]:
        """Get formatted search history for UI display"""
        return self.conv_agent.search_history

    def get_search_by_index(self, index: int) -> Dict:
        """Get specific search result by index"""
        if 0 <= index < len(self.conv_agent.search_history):
            return self.conv_agent.search_history[index]
        return {}

    def format_history_for_display(self) -> List[Dict]:
        """Format search history for UI display with readable timestamps"""
        formatted_history = []
        for i, search in enumerate(self.conv_agent.search_history):
            formatted_history.append({
                "index": i,
                "timestamp": search["timestamp"],
                "readable_time": self._format_timestamp(search["timestamp"]),
                "query": search["query"],
                "results_count": len(search.get("top_docs", [])),
                "preview": f"Found {len(search.get('top_docs', []))} recommendations for '{search['query']}'"
            })
        return formatted_history

    def _format_timestamp(self, iso_timestamp: str) -> str:
        """Convert ISO timestamp to readable format"""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return iso_timestamp


