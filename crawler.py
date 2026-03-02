import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebCrawler:
    """
    A production-ready crawler that performs parallelized depth-limited BFS
    to discover internal links on a target website.
    """
    def __init__(self, base_url: str, max_depth: int = 2, timeout: int = 10, max_workers: int = 5, verify_ssl: bool = True):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.timeout = timeout
        self.max_workers = max_workers
        self.verify_ssl = verify_ssl
        self.visited = set()
        self.site_graph = {} # {url: [child_urls]}

    def is_internal(self, url: str) -> bool:
        """Checks if a URL belongs to the same domain."""
        parsed = urlparse(url)
        return parsed.netloc == '' or parsed.netloc == self.domain

    def normalize_url(self, url: str) -> str:
        """Joins relative URLs and removes fragments."""
        joined = urljoin(self.base_url, url)
        # Remove queries and fragments for normalization
        parsed = urlparse(joined)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
        return normalized

    def get_links(self, url: str) -> list[str]:
        """Fetches a page and extracts all internal links."""
        try:
            response = requests.get(url, timeout=self.timeout, verify=self.verify_ssl)
            response.raise_for_status()
            
            # Only parse HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            for a in soup.find_all('a', href=True):
                raw_href = a['href']
                if self.is_internal(raw_href):
                    normalized = self.normalize_url(raw_href)
                    links.append(normalized)
            return list(set(links)) # Unique links from this page
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

    def crawl(self) -> dict:
        """
        Executes parallel crawl starting from base_url up to max_depth.
        """
        current_layer = {self.normalize_url(self.base_url)}
        self.visited.update(current_layer)

        for depth in range(self.max_depth + 1):
            logger.info(f"Crawling depth {depth}: {len(current_layer)} pages")
            
            if depth >= self.max_depth:
                # Last layer, just initialize their graph entries as empty or known if they were children
                for url in current_layer:
                    if url not in self.site_graph:
                        self.site_graph[url] = []
                break

            next_layer = set()
            
            # Parallel fetch for the current layer
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.get_links, url): url for url in current_layer}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        children = future.result()
                        # Filter children to only unvisited ones for the NEXT layer
                        internal_children = [c for c in children if self.is_internal(c)]
                        self.site_graph[url] = internal_children
                        
                        for child in internal_children:
                            if child not in self.visited:
                                self.visited.add(child)
                                next_layer.add(child)
                    except Exception as exc:
                        logger.error(f"{url} generated an exception: {exc}")
                        self.site_graph[url] = []

            current_layer = next_layer
            if not current_layer:
                break

        return self.site_graph

if __name__ == "__main__":
    crawler = WebCrawler("https://example.com", max_depth=1)
    results = crawler.crawl()
    print(f"Discovered {len(results)} pages.")
