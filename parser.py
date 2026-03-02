import requests
from bs4 import BeautifulSoup, Tag
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SectionParser:
    """
    Parses HTML into hierarchical sections based on H1, H2, and H3 tags.
    """
    def __init__(self):
        self.tags_to_track = ['h1', 'h2', 'h3']

    def parse(self, html: str, url: str) -> List[Dict[str, Any]]:
        """
        Processes HTML content and returns a list of sections.
        Each section has: title, level, content, and source URL.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        page_title = soup.title.string if soup.title else "Untitled Page"
        
        sections = []
        current_section = {
            "title": page_title,
            "level": 0, # Root/Page level
            "content": "",
            "url": url,
            "anchor": ""
        }
        
        # Iterate through content-bearing elements
        main_content = soup.find('body')
        if not main_content:
            logger.warning(f"No body tag found in {url}. Parsing whole document.")
            main_content = soup
        
        tags_to_extract = ['h1', 'h2', 'h3', 'p', 'li', 'span', 'div', 'a', 'header', 'footer', 'nav', 'address', 'td', 'th', 'section', 'article']
        for element in main_content.find_all(tags_to_extract):
            if element.name in self.tags_to_track:
                # Save previous section if it has content or title
                if current_section["content"].strip() or (current_section["level"] == 0 and current_section["title"]):
                    sections.append(current_section)
                
                # Start new section
                try:
                    level = int(element.name[1])
                except (ValueError, IndexError):
                    level = 1
                    
                current_section = {
                    "title": element.get_text(separator=' ', strip=True) or f"Section {level}",
                    "level": level,
                    "content": "",
                    "url": url,
                    "anchor": element.get('id', '')
                }
            else:
                # Append text to current section
                
                # Avoid duplicative text if we're looking at a container that contains block elements we'll see later
                if element.name in ['div', 'header', 'footer', 'nav', 'address', 'section', 'article']:
                    if element.find(['p', 'li', 'h1', 'h2', 'h3', 'table']):
                        continue
                
                text = element.get_text(separator=' ', strip=True)

                if element.name == 'a':
                    href = element.get('href', '')
                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        text = f"{text} (Link: {href})" if text else f"(Link: {href})"

                if text:
                    current_section["content"] += text + " "

        # Final section
        if current_section["content"].strip() or current_section["level"] > 0:
            sections.append(current_section)

        # Post-processing: If level 0 section is empty but we have H-sections, 
        # it might just be the container. 
        return self._clean_sections(sections)

    def _clean_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans whitespace and filters out trivial sections."""
        cleaned = []
        for sec in sections:
            sec["content"] = sec["content"].strip()
            if sec["title"] or sec["content"]:
                cleaned.append(sec)
        return cleaned

if __name__ == "__main__":
    # Test sample
    test_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>Intro text about the tool.</p>
            <h2>Section 1</h2>
            <p>Details about section 1.</p>
            <h3>Subsection 1.1</h3>
            <p>More granular info.</p>
            <h2>Section 2</h2>
            <p>End of page.</p>
        </body>
    </html>
    """
    parser = SectionParser()
    sections = parser.parse(test_html, "https://example.com/test")
    for s in sections:
        print(f"[{s['level']}] {s['title']}: {len(s['content'])} chars")
