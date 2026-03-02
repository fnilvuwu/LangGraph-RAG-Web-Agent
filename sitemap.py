from urllib.parse import urlparse
from typing import Dict, List, Any
import pyvis
from pyvis.network import Network
import tempfile

class SitemapVisualizer:
    """
    Converts a flat site graph {url: [children]} into a nested tree structure
    and provides helpers for Streamlit visualization.
    """
    def __init__(self, site_graph: Dict[str, List[str]]):
        self.site_graph = site_graph

    def generate_pyvis_html(self) -> str:
        """
        Generates an interactive HTML representation of the site graph using PyVis.
        Returns the HTML string for embedding.
        """
        # Create a pyvis network
        net = Network(height="600px", width="100%", bgcolor="#0f172a", font_color="white", directed=True)
        # Add physics for nice animation
        net.barnes_hut(gravity=-3000, central_gravity=0.1, spring_length=150)
        
        # Add nodes and edges
        nodes = set()
        edges = []
        for parent, children in self.site_graph.items():
            parent_name = urlparse(parent).path or "/"
            nodes.add((parent, parent_name))
            for child in children:
                child_name = urlparse(child).path or "/"
                nodes.add((child, child_name))
                edges.append((parent, child))
                
        for node_id, node_label in nodes:
            # Highlight root node differently?
            color = "#3b82f6" # primary blue
            if node_label == "/":
                color = "#fde047" # highlight yellow
            elif node_label.startswith("http"):
                 color = "#fde047"
            net.add_node(node_id, label=node_label, title=node_id, color=color)
            
        for src, dst in edges:
            net.add_edge(src, dst, color="#334155")
            
        # Generate HTML string
        return net.generate_html()

    def build_tree(self) -> Dict[str, Any]:
        """
        Transforms the flat graph into a nested tree starting from the root URL.
        """
        urls = sorted(self.site_graph.keys())
        if not urls:
            return {}
        
        root_url = urls[0] # Assumes the first crawled is the root
        return self._build_recursive(root_url, set())

    def _build_recursive(self, url: str, visited: set) -> Dict[str, Any]:
        if url in visited:
            return {"url": url, "children": []}
        
        visited.add(url)
        children = self.site_graph.get(url, [])
        
        tree = {
            "url": url,
            "name": urlparse(url).path or "/",
            "children": [self._build_recursive(child, visited) for child in children]
        }
        return tree

    @staticmethod
    def render_markdown_tree(tree: Dict[str, Any], level: int = 0) -> str:
        """
        Generates a markdown string representing the tree hierarchy.
        """
        indent = "  " * level
        md = f"{indent}- [{tree['name']}]({tree['url']})\n"
        for child in tree.get("children", []):
            md += SitemapVisualizer.render_markdown_tree(child, level + 1)
        return md

if __name__ == "__main__":
    # Test logic
    sample_graph = {
        "https://example.com/": ["https://example.com/about", "https://example.com/contact"],
        "https://example.com/about": ["https://example.com/team"],
        "https://example.com/contact": [],
        "https://example.com/team": []
    }
    viz = SitemapVisualizer(sample_graph)
    tree = viz.build_tree()
    print(viz.render_markdown_tree(tree))
