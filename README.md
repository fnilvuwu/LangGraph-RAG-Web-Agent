# 🌐 LangGraph RAG Web Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![LLM Integration](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI%20%7C%20OpenRouter-purple)

**LangGraph RAG Web Agent** is an intelligent, agentic web navigation and intelligence extraction application. Built using modern RAG (Retrieval-Augmented Generation) architectures and LangGraph, it empowers users to crawl, semantic-search, and chat with entire websites effortlessly.

Whether it's dynamically extracting contact information, summarizing key services, or pulling deeply nested pricing data, this tool transforms the static web into an interactive, queryable database.

---

## ✨ Key Features

- **🕸️ Intelligent Web Crawling & Parsing**: Automatically traverses websites based on user-defined parameters (e.g., Target URL, Crawl Depth) and extracts text structure, capturing semantic boundaries (headers, paragraphs, lists) out of raw HTML.
- **🗺️ Interactive Sitemap Visualization**: Generates a visually appealing, interactive graph of the crawled website architecture using `pyvis`, giving users immediate insight into the site's structure.
- **🧠 Advanced RAG Architecture**: Uses a hierarchical embedding approach. Instead of chunking blindly, it embeds content contextually, preserving the hierarchy of web pages for high-accuracy retrieval.
- **🛠️ Agentic Capabilities (LangGraph)**: The `WebNavigatorAgent` determines when to read more pages, when to use the sitemap tool, and when to synthesize an answer. It acts autonomously to fulfill user prompts like "find contact info."
- **🔄 Multi-LLM Provider Support**: Flexible architecture supporting API keys from Google (Gemini), OpenAI, and OpenRouter, along with Local (HuggingFace) embedding fallback to minimize costs.
- **💻 Glassmorphism UI**: A sleek, premium, and responsive user interface built in Streamlit, featuring an interactive agent chat, a page explorer, and one-click quick action buttons.

---

## 🏗️ Architecture & Tech Stack

This project demonstrates strong software engineering patterns, modularity, and a deep understanding of Generative AI integration.

*   **Frontend**: Streamlit (with custom CSS for a premium UI)
*   **Orchestration & Agent**: LangChain and LangGraph
*   **Embeddings & Vector Store**: Chromadb (via LangChain integrations)
*   **LLM Providers**: `google-generativeai`, `openai`, `langchain-anthropic` (via OpenRouter)
*   **Web Scraping & Parsing**: `requests`, `BeautifulSoup4`
*   **Visualization**: `pyvis` (Interactive HTML network graphs)

### Core Modules
*   `app.py`: The Streamlit entry point, managing UI state, authentication, and layouts.
*   `agent.py`: Implementing the LangGraph-based Tool Calling Agent that interprets queries and executes tools.
*   `crawler.py`: Handles network requests, honoring crawl depths, and parsing HTML with `SectionParser`.
*   `embeddings.py` & `retriever.py`: Manages vectorization of text chunks and handles hierarchical semantic search.
*   `sitemap.py`: Converts crawl graphs into visual network representations.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   API Keys for your preferred LLM provider (Gemini, OpenAI, or OpenRouter).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/LangGraph-RAG-Web-Agent.git
    cd LangGraph-RAG-Web-Agent
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration (Optional):**
    You can provide a `.env` file at the root of the project with default keys, but the application also supports secure API key entry directly through the sidebar UI at runtime.
    ```env
    GOOGLE_API_KEY="your-gemini-key"
    ```

### Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 💡 Usage Guide

1. **Authenticate**: Open the sidebar and select your preferred LLM and Embedding provider. Enter the required API key(s).
2. **Crawl a Website**: Enter a target URL (e.g., `https://example.com`) and choose a crawl depth. Click **"Start Crawling"**.
3. **Explore**:
   *   **Interactive Sitemap**: View the structure of the scanned web pages visually.
   *   **Page Explorer**: Browse through individually extracted sections and headings of pulled pages.
4. **Chat & Action**: Use the Agent Chat to ask natural language questions about the site. Use the Quick Action buttons to instantly process repetitive tasks (e.g., Extracting Pricing, Finding Contact Info).

---

## 👨‍💻 Note for Recruiters & Stakeholders

This application was built to showcase the capability of integrating modern Large Language Models within standard Software Engineering practices. It highlights:
- **Systematic Problem Solving**: Breaking down web crawling into autonomous agent tools.
- **UX/UI Implementation**: Building user-friendly interfaces around complex AI concepts, managing loading states, and handling streaming interactions gracefully in Python.
- **Adaptability**: Allowing immediate swapping between Open-Source and Proprietary foundational models.
- **Data Engineering**: Processing raw, messy HTML into clean, semantically chunked markdown documents for Vector Storage.

Expect high maintainability, documented code, and an architecture ready to be extended with further tools (e.g., automated form-filling, scheduled monitoring).
