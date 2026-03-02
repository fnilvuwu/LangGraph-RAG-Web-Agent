import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalEmbedder:
    """
    Manages two levels of embeddings:
    1. Page-level (Summaries)
    2. Section-level (H1-H3 sections)
    Uses Google Gemini for speed and performance.
    """
    def __init__(self, persist_directory: str = "./chroma_db", emb_provider: str = "Gemini", api_key: str = None):
        self.persist_directory = persist_directory
        self.api_key = api_key
        
        if emb_provider == "Gemini":
            self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not found. Ensure it is set.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key, model="models/gemini-embedding-001")
        elif emb_provider == "OpenAI":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found. Ensure it is set.")
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key, model="text-embedding-3-small")
        elif emb_provider == "HuggingFace":
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown Embedding Provider: {emb_provider}")

        # Initialize collections
        self.section_store = Chroma(
            collection_name="sections",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(persist_directory, "sections")
        )
        
        self.page_store = Chroma(
            collection_name="pages",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(persist_directory, "pages")
        )

    def generate_page_summary(self, sections: List[Dict[str, Any]]) -> str:
        """Generates a concise summary for a page based on its sections."""
        if not sections:
            return "Empty page."

        # Replaced LLM summarization with heuristic extraction to save API quota
        summary_lines = []
        for s in sections[:3]:  # Take content from first few sections
            content = s['content'][:200].strip()  # Take first 200 chars
            if content:
                summary_lines.append(f"{s['title']}: {content}...")
        
        if not summary_lines:
            return "Summary unavailable (heuristic extraction active)."
        
        return " | ".join(summary_lines)

    def add_page(self, url: str, sections: List[Dict[str, Any]]):
        """Processes a page: generates summary and embeds all levels."""
        logger.info(f"Processing embeddings for {url}")
        
        # 1. Store Sections
        section_docs = []
        for sec in sections:
            metadata = {
                "url": url,
                "title": sec["title"],
                "level": sec["level"],
                "anchor": sec.get("anchor", ""),
                "type": "section"
            }
            doc = Document(page_content=sec["content"], metadata=metadata)
            section_docs.append(doc)
        
        if section_docs:
            self.section_store.add_documents(section_docs)
            logger.info(f"Added {len(section_docs)} sections to core index.")

        # 2. Generate and Store Summary
        summary = self.generate_page_summary(sections)
        summary_metadata = {
            "url": url,
            "title": sections[0]["title"] if sections else "Untitled",
            "type": "page_summary"
        }
        self.page_store.add_documents([Document(page_content=summary, metadata=summary_metadata)])
        logger.info(f"Added page summary for {url}")

    def similarity_search_sections(self, query: str, k: int = 4) -> List[Document]:
        """Search specifically at the section level."""
        return self.section_store.similarity_search(query, k=k)

    def similarity_search_pages(self, query: str, k: int = 3) -> List[Document]:
        """Search at the page summary level."""
        return self.page_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Smoke test structure
    embedder = HierarchicalEmbedder(persist_directory="./test_db")
    mock_sections = [
        {"title": "Introduction", "content": "This is a web navigator tool.", "level": 1},
        {"title": "Features", "content": "It crawls and parses hierarchy.", "level": 2}
    ]
