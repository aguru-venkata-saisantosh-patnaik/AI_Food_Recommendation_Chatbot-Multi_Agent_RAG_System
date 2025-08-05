# Cell: Sharded Retrieval Agent (Modified for LLM Integration)

import pandas as pd
import json
import os
import logging
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class ShardedRetrievalAgent:
    """
    Main retrieval agent that searches across multiple ChromaDB shards.
    Modified to work with LLM-refined queries and filters from OpenAIQueryEnhancer.
    """

    def __init__(self, shard_info_path, embeddings):
        self.shard_info_path = shard_info_path
        self.embeddings = embeddings
        self.shards_df = pd.read_csv(shard_info_path)

    def search_shard(self, shard_meta, query, chroma_filter, top_k=5):
        """Search a single shard with given query and filter"""
        collection_path = shard_meta['persist_directory']

        if not os.path.exists(collection_path):
            raise FileNotFoundError(f"Collection path does not exist: {collection_path}")

        vectordb = Chroma(
            persist_directory=collection_path,
            collection_name=shard_meta['collection_name'],
            embedding_function=self.embeddings,
        )

        # Handle filter - convert "NO_FILTER" string to None
        filter_to_use = None if chroma_filter == "NO_FILTER" else chroma_filter

        try:
            results = vectordb.similarity_search(
                query=query,
                k=top_k,
                filter=filter_to_use
            )
            return results
        except Exception as e:
            logger.error(f"Error searching shard {shard_meta['collection_name']}: {e}")
            return []

    def gather_shard_results(self, query, chroma_filter, top_k=5):
        """Gather results from all shards"""
        all_results = []

        for idx, shard_meta in self.shards_df.iterrows():
            try:
                result = self.search_shard(shard_meta, query, chroma_filter, top_k)
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to search shard {idx}: {e}")
                continue

        # Flatten results from all shards
        flattened_results = [doc for result in all_results for doc in result]
        return flattened_results

    def retrieve_with_refined_query(self, refined_query, refined_filter, top_k_per_shard=5):
        """
        Retrieve documents using already refined query and filter from LLM.
        This replaces the old retrieve method that used CustomSelfQueryConstructor.
        """
        all_docs = self.gather_shard_results(refined_query, refined_filter, top_k_per_shard)

        print(f"📊 Total docs gathered from all shards: {len(all_docs)}")
        print(f"🔍 Used query: '{refined_query}'")
        print(f"🔧 Used filter: {refined_filter}")

        return all_docs

    def get_all_docs_formatted(self, refined_query, refined_filter, top_k_per_shard=5):
        """
        Get all documents in formatted structure for downstream processing.
        This replaces the old get_all_docs method.
        """
        all_shard_docs = self.retrieve_with_refined_query(
            refined_query, refined_filter, top_k_per_shard
        )

        # Format documents for consistency with existing pipeline
        formatted_data = []
        for doc in all_shard_docs:
            formatted_data.append({
                "id": doc.id if hasattr(doc, 'id') else None,
                "page_content": doc.page_content if hasattr(doc, 'page_content') else "",
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })

        return formatted_data


def retrieve_all_docs_with_llm_query(refined_query, refined_filter, shard_info_path,
                                   embeddings, top_k_per_shard=5):
    """
    Main function to retrieve documents using LLM-refined query and filter.
    This replaces the old retrieve_all_docs function.
    """
    # Create agent instance
    agent = ShardedRetrievalAgent(shard_info_path, embeddings)

    # Get formatted documents
    formatted_docs = agent.get_all_docs_formatted(
        refined_query, refined_filter, top_k_per_shard
    )

    return formatted_docs

print("✅ Modified ShardedRetrievalAgent implemented")
print("🔧 Integrated with LLM query refinement from OpenAIQueryEnhancer")
print("📦 Ready for use with refined queries and filters")
