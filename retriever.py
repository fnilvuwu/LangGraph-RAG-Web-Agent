import logging
from typing import List, Dict, Any
from embeddings import HierarchicalEmbedder
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalRetriever:
    """
    Implements a two-stage retrieval pipeline:
    1. Search page summaries to narrow down candidate pages.
    2. Search specific sections within those pages.
    """
    def __init__(self, embedder: HierarchicalEmbedder):
        self.embedder = embedder

    def retrieve(self, query: str, top_k_pages: int = 3, top_k_sections: int = 5) -> List[Dict[str, Any]]:
        """
        Performs the hierarchical search.
        """
        logger.info(f"Initiating two-stage retrieval for: '{query}'")

        # Stage 1: Search page summaries
        relevant_pages = self.embedder.similarity_search_pages(query, k=top_k_pages)
        if not relevant_pages:
            logger.warning("No relevant pages found in Stage 1.")
            return []

        # Extract URLs of relevant pages
        target_urls = list(set([doc.metadata.get("url") for doc in relevant_pages if "url" in doc.metadata]))
        logger.info(f"Stage 1 found {len(target_urls)} candidate pages.")

        # Stage 2: Search sections within these pages
        # Chroma supports metadata filtering
        filter_dict = {"url": {"$in": target_urls}}
        
        # We use the underlying section_store directly for filtered search
        section_results = self.embedder.section_store.similarity_search(
            query, 
            k=top_k_sections,
            filter=filter_dict
        )

        logger.info(f"Stage 2 found {len(section_results)} matching sections.")

        # Format results for the agent/UI
        formatted_results = []
        for doc in section_results:
            formatted_results.append({
                "content": doc.page_content,
                "title": doc.metadata.get("title", "Untitled Section"),
                "url": doc.metadata.get("url"),
                "level": doc.metadata.get("level"),
                "anchor": doc.metadata.get("anchor", ""),
                "score": 1.0 # Placeholder for similarity score if needed
            })

        return formatted_results

if __name__ == "__main__":
    # Integration layout
    # embedder = HierarchicalEmbedder()
    # retriever = HierarchicalRetriever(embedder)
    # results = retriever.retrieve("How to install?")
    pass
