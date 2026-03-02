import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# LangChain v1.x uses create_agent from langchain.agents
from langchain.agents import create_agent

from embeddings import HierarchicalEmbedder
from retriever import HierarchicalRetriever
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI

class WebNavigatorAgent:
    """
    An agentic layer over the hierarchical retrieval system.
    Uses configurable LLM backends (Gemini, OpenAI, OpenRouter) for reasoning and tool orchestration.
    """
    def __init__(self, embedder: HierarchicalEmbedder, retriever: HierarchicalRetriever, llm_provider: str = "Gemini", llm_model: str = "gemini-2.5-flash", api_key: str = None):
        self.embedder = embedder
        self.retriever = retriever
        self.api_key = api_key
        
        # Initialize LLM backend
        if llm_provider == "Gemini":
            self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0, google_api_key=self.api_key)
        elif llm_provider == "OpenAI":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(model=llm_model, temperature=0, openai_api_key=self.api_key)
        elif llm_provider == "OpenRouter":
            self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
            self.llm = ChatOpenAI(
                model=llm_model, 
                temperature=0, 
                openai_api_key=self.api_key, 
                openai_api_base="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unknown LLM Provider: {llm_provider}")
            
        self.agent_graph = self._setup_agent()

    def _setup_agent(self):
        @tool
        def get_sitemap() -> str:
            """Returns the list of all available URLs discovered on the site."""
            if not getattr(self, "site_graph", None):
                return "The sitemap has not been built yet. I cannot browse until the user crawls a URL first."
            urls = list(self.site_graph.keys())
            if not urls:
                return "No URLs found in the sitemap."
            return "Available URLs:\n" + "\n".join(f"- {u}" for u in urls)

        @tool
        def search_sections(query: str, urls: List[str]) -> str:
            """Searches for specific sections within a list of targeted URLs. Ensure the URLs have been read via read_page first."""
            if isinstance(urls, str):
                urls = [urls]
            
            filter_dict = {"url": {"$in": urls}}
            docs = self.embedder.section_store.similarity_search(query, k=5, filter=filter_dict)
            
            if not docs:
                return "No matching sections found in these pages."
            
            results = []
            for d in docs:
                results.append(f"PAGE: {d.metadata.get('url')}\nSECTION: {d.metadata.get('title')}\nCONTENT: {d.page_content}\n---")
            return "\n".join(results)

        @tool
        def read_page(url: str) -> str:
            """Fetches a page, reads its contents, and provides a summary and section headers. ALWAYS call this tool to learn about a new page."""
            import requests
            from parser import SectionParser
            
            # Check if already embedded
            page_docs = self.embedder.page_store.get(where={"url": url})
            if not page_docs or not page_docs.get('documents'):
                try:
                    verify_ssl = getattr(self, "verify_ssl", True)
                    resp = requests.get(url, timeout=10, verify=verify_ssl)
                    if 'text/html' in resp.headers.get('Content-Type', '').lower():
                        parser = SectionParser()
                        sections = parser.parse(resp.text, url)
                        self.embedder.add_page(url, sections)
                    else:
                        return f"URL {url} is not an HTML page."
                except Exception as e:
                    return f"Error fetching {url}: {e}"
            
            # Fetch summary and headers from DB
            page_docs = self.embedder.page_store.get(where={"url": url})
            summary = page_docs['documents'][0] if page_docs and page_docs.get('documents') else "No summary available."
            
            section_docs = self.embedder.section_store.get(where={"url": url})
            headers = []
            if section_docs and 'metadatas' in section_docs:
                for meta in section_docs['metadatas']:
                    if meta.get('type') == 'section':
                        headers.append(f"- {meta.get('title')} (Level {meta.get('level')})")
            
            return f"SUMMARY: {summary}\n\nAVAILABLE SECTIONS (can be searched via search_sections):\n" + "\n".join(headers)

        tools = [get_sitemap, search_sections, read_page]
        
        system_prompt = (
            "You are an Agentic RAG Web Explorer. Your goal is to help users find information within a crawled website using Retrieval-Augmented Generation. "
            "You simulate browsing by using tools: "
            "1. Use `get_sitemap` to see the structure of the site and identify relevant URLs. "
            "2. Use `read_page` to actually visit a page, which fetches it and gives you a summary and available sections. "
            "3. Use `search_sections` if you need detailed information from specifically targeted pages after reading them. "
            "Always cite your sources (URLs and Section Titles). "
            "NEVER hallucinate or guess information such as emails, phone numbers, or names. Only provide information strictly found in tool outputs. "
            "If you do not have a specific URL to search, ALWAYS use `get_sitemap` first. DO NOT ask the user to provide a URL."
        )

        # In LangChain v1.x, create_agent returns a CompiledStateGraph
        return create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def ask(self, query: str, callbacks=None):
        """Runs the agent reasoning loop."""
        config = {}
        if callbacks:
            config["callbacks"] = callbacks
            
        # Invoking the graph with messages state
        result = self.agent_graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        
        # The result is the final state dictionary.
        # We want the content of the last AIMessage.
        messages = result.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            return {"output": messages[-1].content}
        
        return {"output": "I'm sorry, I couldn't generate a response."}
