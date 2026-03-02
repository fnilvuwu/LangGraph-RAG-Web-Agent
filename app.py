import streamlit as st
import os
import requests
from crawler import WebCrawler
from parser import SectionParser
from embeddings import HierarchicalEmbedder
from retriever import HierarchicalRetriever
from agent import WebNavigatorAgent
from sitemap import SitemapVisualizer
import time
from dotenv import load_dotenv
import threading
# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="LangGraph RAG Web Agent", layout="wide", page_icon="🌐")

# Custom CSS for glassmorphism and premium look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
    }
    .sidebar .sidebar-content {
        background-color: #0f172a;
    }
    .highlight {
        background-color: #fde047;
        color: black;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .social-links {
        display: flex;
        gap: 1.2rem;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #334155;
    }
    .social-links a {
        color: #94a3b8;
        transition: color 0.3s ease;
    }
    .social-links a.linkedin:hover { color: #22d3ee; }
    .social-links a.instagram:hover { color: #f472b6; }
    .social-links a.youtube:hover { color: #ef4444; }
    .social-links a.fiverr {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-decoration: none;
    }
    .social-links a.fiverr:hover { color: #4ade80; }
    .social-links svg {
        height: 1.25rem;
        width: 1.25rem;
        fill: currentColor;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: AUTHENTICATION ---
with st.sidebar:
    st.title("🌐 LangGraph RAG Web Agent")
    st.markdown("---")
    
    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI", "OpenRouter"])
    
    if llm_provider == "Gemini":
        llm_api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="Enter Google API Key")
        llm_model = st.text_input("Model Name", value="gemini-2.5-flash")
    elif llm_provider == "OpenAI":
        llm_api_key = st.text_input("🔑 OpenAI API Key", type="password", placeholder="Enter OpenAI API Key")
        llm_model = st.text_input("Model Name", value="gpt-4o-mini")
    else:
        llm_api_key = st.text_input("🔑 OpenRouter API Key", type="password", placeholder="Enter OpenRouter API Key")
        llm_model = st.text_input("Model Name", value="anthropic/claude-3-haiku")
        
    st.markdown("### 🧠 Embeddings Settings")
    emb_provider = st.selectbox("Embedding Provider", ["Gemini", "OpenAI", "HuggingFace (Local)"])
    
    if emb_provider == "Gemini":
        emb_api_key = st.text_input("🔑 Gemini API Key for Embeddings", type="password", value=llm_api_key if llm_provider=="Gemini" else "")
    elif emb_provider == "OpenAI":
        emb_api_key = st.text_input("🔑 OpenAI API Key for Embeddings", type="password", value=llm_api_key if llm_provider=="OpenAI" else "")
    else:
        emb_api_key = None
        st.info("Local embeddings run on your CPU/GPU (requires `sentence-transformers`).")

    if not llm_api_key:
        st.warning("Please enter your LLM API Key.")
        st.stop()
        
    if emb_provider != "HuggingFace (Local)" and not emb_api_key:
        st.warning("Please enter your Embedding API Key.")
        st.stop()
        
    # Apply to environment variables if needed
    if llm_provider == "Gemini" or emb_provider == "Gemini":
        os.environ["GOOGLE_API_KEY"] = llm_api_key if llm_provider == "Gemini" else emb_api_key
    if llm_provider == "OpenAI" or emb_provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = llm_api_key if llm_provider == "OpenAI" else emb_api_key
    if llm_provider == "OpenRouter":
        os.environ["OPENROUTER_API_KEY"] = llm_api_key

# --- STATE INITIALIZATION ---
# Check if config changed
current_config = f"{llm_provider}_{llm_model}_{emb_provider}_{llm_api_key[:5] if llm_api_key else ''}_{emb_api_key[:5] if emb_api_key else ''}"
if "app_config" not in st.session_state or st.session_state.app_config != current_config:
    st.session_state.app_config = current_config
    
    db_dir = f"./web_nav_db_{emb_provider.split(' ')[0]}"
    try:
        provider_name = "HuggingFace" if emb_provider == "HuggingFace (Local)" else emb_provider
        st.session_state.embedder = HierarchicalEmbedder(persist_directory=db_dir, emb_provider=provider_name, api_key=emb_api_key)
        st.session_state.retriever = HierarchicalRetriever(st.session_state.embedder)
        st.session_state.agent = WebNavigatorAgent(
            st.session_state.embedder, 
            st.session_state.retriever,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=llm_api_key
        )
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        st.stop()

if "site_graph" not in st.session_state:
    st.session_state.site_graph = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "crawler_inst" not in st.session_state:
    st.session_state.crawler_inst = None

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    target_url = st.text_input("🔗 Target URL", placeholder="https://example.com")
    max_depth = st.slider("Crawl Depth", 1, 3, 1)
    verify_ssl = st.checkbox("Verify SSL", value=True, help="Disable if you get SSL certificate errors")
    
    if st.button("🚀 Start Crawling"):
        if target_url:
            with st.status("🏗️ Building Sitemap...") as status:
                st.write("🕷️ Discovering links...")
                crawler = WebCrawler(target_url, max_depth=max_depth, verify_ssl=verify_ssl)
                graph = crawler.crawl()
                st.session_state.site_graph = graph
                
                # Pass site_graph to agent so it can do lazy fetching
                st.session_state.agent.site_graph = graph
                st.session_state.agent.verify_ssl = verify_ssl
                
                status.update(label="✅ Sitemap Complete!", state="complete")
        else:
            st.warning("Please enter a URL first.")

    st.markdown("### 🗺️ Sitemap")
    if st.session_state.site_graph:
        st.success(f"{len(st.session_state.site_graph)} pages mapped. Explore them in the tabs.")
    else:
        st.info("Crawl a site to see its structure.")

    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <p style="font-size: 0.9rem; color: #94a3b8;">Built by Fnilvuwu</p>
        </div>  
        <div class="social-links">
            <a href="https://linkedin.com/in/fnilvuwu" target="_blank" rel="noreferrer" class="linkedin" title="LinkedIn">
                <svg viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" /></svg>
            </a>
            <a href="https://instagram.com/fnilvuwu" target="_blank" rel="noreferrer" class="instagram" title="Instagram">
                <svg viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" /></svg>
            </a>
            <a href="https://youtube.com/@fnilvuwu" target="_blank" rel="noreferrer" class="youtube" title="YouTube">
                <svg viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" /></svg>
            </a>
            <a href="https://fiverr.com/fnilvuwu" target="_blank" rel="noreferrer" class="fiverr" title="Fiverr">
                Fiverr
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.header("🧠 Intelligent Browsing")

tab_chat, tab_preview, tab_sitemap = st.tabs(["💬 Agent Chat", "📄 Page Explorer", "🕸️ Interactive Sitemap"])

with tab_chat:
    # Chat Container
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Quick Action Buttons
    st.markdown("<div style='text-align: center'><b>Quick Actions:</b></div>", unsafe_allow_html=True)
    st.write("")
    _, m1, m2, m3, _ = st.columns([1, 2, 2, 2, 1])
    magic_prompt = None
    if m1.button("💰 Extract pricing", use_container_width=True):
        magic_prompt = "Extract all product pricing data from this site. Use the `get_sitemap` tool to find pages likely to contain pricing (e.g., Pricing, Products, Store), then use `read_page` (MAXIMUM 2 pages to save API quota) and `search_sections` to extract the data. Format as a table if possible. DO NOT ask the user for a URL."
    if m2.button("📞 Find contact info", use_container_width=True):
        magic_prompt = "Extract all contact information (emails, phone numbers, social links, addresses) from the site. Use the `get_sitemap` tool to find Contact or About pages, then `read_page` (MAXIMUM 2 pages to save API quota) and `search_sections` to extract the exact details without hallucinating. DO NOT ask the user for a URL. Format the output as a clear readable list."
    if m3.button("📝 Summarize site", use_container_width=True):
        magic_prompt = "Summarize the main features or services offered by this site. Start by using `get_sitemap` to understand the site structure, then use `read_page` (MAXIMUM 2 pages to save API quota) on the main pages (like Home or About). DO NOT ask the user for a URL."

    # User Input
    prompt = st.chat_input("Ask about the website...")
    
    # Process prompt from either input or magic button
    active_prompt = prompt or magic_prompt
    
    if active_prompt:
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(active_prompt)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🤔 Thinking..."):
                        response = st.session_state.agent.ask(active_prompt)
                    output = response["output"]
                    st.markdown(output)
                    st.session_state.chat_history.append({"role": "assistant", "content": output})
                except Exception as e:
                    st.error(f"Agent Error: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        chat_text = ""
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Agent"
            chat_text += f"{role}:\n{msg['content']}\n\n---\n\n"
            
        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="agent_chat_history.md",
            mime="text/markdown"
        )

with tab_preview:
    st.markdown("### 📄 Page Explorer")
    if not st.session_state.site_graph:
        st.info("Please start crawling a website from the sidebar first to explore its pages.")
    else:
        preview_url = st.selectbox("Select Page to Preview", options=list(st.session_state.site_graph.keys()))
        
        if preview_url:
            # Reorganize into a sidebar-like navigation (left) and content area (right)
            nav_col, content_col = st.columns([1, 3])
            
            with nav_col:
                st.markdown("#### 🧭 Contents")
                
                # Render Search in the navigation column
                query = st.text_input("Search in page...", key="page_search")
                if query:
                    st.markdown("**Search Results**")
                    results = st.session_state.embedder.section_store.similarity_search(
                        query, k=3, filter={"url": preview_url}
                    )
                    for res in results:
                        st.info(f"**{res.metadata.get('title')}**\n{res.page_content[:100]}...")
                
                st.markdown("---")
                
                # Fetch sections
                section_docs = st.session_state.embedder.section_store.get(where={"url": preview_url})
                selected_idx = None
                
                if section_docs and section_docs['documents']:
                    # Create a navigable list of sections mapped to their depth
                    def format_section_name(i):
                        meta = section_docs['metadatas'][i]
                        level = int(meta.get('level', 1))
                        # Create structural indentation
                        indent = "└ " if level > 1 else ""
                        indent += "—" * (level - 1)
                        if indent: indent += " "
                        return f"{indent}{meta.get('title', 'Untitled')}"

                    selected_idx = st.radio(
                        "Sections", 
                        options=range(len(section_docs['documents'])),
                        format_func=format_section_name,
                        label_visibility="collapsed"
                    )
                else:
                    st.warning("No sections mapped.")

            with content_col:
                st.markdown(f"**URL:** [{preview_url}]({preview_url})")
                
                # Try to find the page summary in the store
                page_docs = st.session_state.embedder.page_store.get(where={"url": preview_url})
                if page_docs and page_docs['documents']:
                    with st.expander("📝 Page Summary", expanded=False):
                        st.info(page_docs['documents'][0])
                
                # Display the selected section
                if section_docs and section_docs['documents'] and selected_idx is not None:
                    sel_doc = section_docs['documents'][selected_idx]
                    sel_meta = section_docs['metadatas'][selected_idx]
                    
                    st.markdown("---")
                    # Add a nice container for the content
                    with st.container(border=True):
                        st.markdown(f"### {sel_meta.get('title', 'Section')}")
                        st.caption(f"Header Level: {sel_meta.get('level', '?')}")
                        st.markdown(sel_doc)
                elif not section_docs or not section_docs['documents']:
                    st.info("Crawl the page to extract section contents.")

with tab_sitemap:
    st.markdown("### 🕸️ Interactive Sitemap")
    if not st.session_state.site_graph:
        st.info("Please start crawling a website from the sidebar first to see the interactive sitemap.")
    else:
        viz = SitemapVisualizer(st.session_state.site_graph)
        try:
            html = viz.generate_pyvis_html()
            st.components.v1.html(html, height=650, scrolling=True)
        except Exception as e:
            st.error(f"Error generating graph: {e}")

