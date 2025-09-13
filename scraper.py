#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AtlanDocScraper:
    def __init__(self):
        self.session = None
        self.scraped_urls = set()
        self.knowledge_base = []
        self.base_urls = {
            "docs": "https://docs.atlan.com/",
            "developer": "https://developer.atlan.com/"
        }
        self.max_pages_per_site = 50  
        self.delay_between_requests = 1
        
    async def create_session(self):
        """Create an aiohttp session with proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common navigation elements
        text = re.sub(r'(Home|Navigation|Menu|Footer|Header|Sidebar)', '', text, flags=re.IGNORECASE)
        
        # Remove very short content
        if len(text) < 50:
            return ""
            
        return text
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML, focusing on documentation"""
        
        # Try to find main content areas
        content_selectors = [
            'main',
            'article', 
            '.content',
            '.main-content',
            '.documentation',
            '.docs-content',
            '#content',
            '.markdown-body',
            '.prose'
        ]
        
        main_content = ""
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text(separator=' ', strip=True)
                break
        
        # Fallback: get all text but filter out navigation
        if not main_content:
            # Remove navigation, footer, header elements
            for tag in soup.find_all(['nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            main_content = soup.get_text(separator=' ', strip=True)
        
        return self.clean_text(main_content)
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant internal links from the page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if urlparse(full_url).netloc in [urlparse(url).netloc for url in self.base_urls.values()]:
                # Filter out non-documentation links
                if not any(skip in full_url.lower() for skip in ['#', 'mailto:', 'tel:', 'javascript:']):
                    links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    async def scrape_page(self, url: str) -> Dict:
        """Scrape a single page and extract content"""
        if url in self.scraped_urls:
            return None
        
        try:
            logger.info(f"Scraping: {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract metadata
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                # Extract main content
                content = self.extract_main_content(soup)
                
                if not content:
                    logger.warning(f"No content extracted from {url}")
                    return None
                
                # Extract links for further crawling
                links = self.extract_links(soup, url)
                
                self.scraped_urls.add(url)
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'links': links,
                    'timestamp': time.time(),
                    'source': 'docs' if 'docs.atlan.com' in url else 'developer'
                }
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    async def crawl_site(self, base_url: str, max_pages: int = 50) -> List[Dict]:
        """Crawl a site starting from base URL"""
        pages_data = []
        urls_to_visit = [base_url]
        visited = set()
        
        while urls_to_visit and len(pages_data) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            # Scrape the page
            page_data = await self.scrape_page(current_url)
            
            if page_data:
                pages_data.append(page_data)
                
                # Add new links to visit (limit to avoid infinite crawling)
                new_links = [link for link in page_data['links'] 
                           if link not in visited and link not in urls_to_visit]
                urls_to_visit.extend(new_links[:10])  # Limit new links per page
            
            # Be respectful - add delay between requests
            await asyncio.sleep(self.delay_between_requests)
        
        return pages_data
    
    async def scrape_all_sites(self) -> List[Dict]:
        """Scrape all configured sites"""
        await self.create_session()
        
        try:
            all_pages = []
            
            for site_name, base_url in self.base_urls.items():
                logger.info(f"Starting to crawl {site_name}: {base_url}")
                site_pages = await self.crawl_site(base_url, self.max_pages_per_site)
                all_pages.extend(site_pages)
                logger.info(f"Scraped {len(site_pages)} pages from {site_name}")
                
                # Delay between sites
                await asyncio.sleep(2)
            
            self.knowledge_base = all_pages
            return all_pages
            
        finally:
            await self.close_session()
    
    def save_knowledge_base(self, filename: str = "atlan_knowledge_base.json"):
        """Save the scraped knowledge base to a JSON file"""
        output_path = Path(filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base saved to {output_path}")
        logger.info(f"Total pages: {len(self.knowledge_base)}")
        
        # Print summary statistics
        source_counts = {}
        for page in self.knowledge_base:
            source = page.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"Pages by source: {source_counts}")
    
    def load_knowledge_base(self, filename: str = "atlan_knowledge_base.json") -> List[Dict]:
        """Load existing knowledge base from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"Loaded {len(self.knowledge_base)} pages from {filename}")
            return self.knowledge_base
        except FileNotFoundError:
            logger.warning(f"Knowledge base file {filename} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return []

async def main():
    """Main function to run the scraper"""
    scraper = AtlanDocScraper()
    
    print("🕷️  Starting Atlan Documentation Scraper...")
    print("=" * 50)
    
    # Check if knowledge base already exists
    existing_kb = scraper.load_knowledge_base()
    
    if existing_kb:
        print(f"📚 Found existing knowledge base with {len(existing_kb)} pages")
        response = input("Do you want to re-scrape? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing knowledge base")
            return
    
    print("🚀 Starting web scraping...")
    print("⏱️  This may take several minutes...")
    
    start_time = time.time()
    
    try:
        pages = await scraper.scrape_all_sites()
        scraper.save_knowledge_base()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Scraping completed!")
        print(f"📊 Statistics:")
        print(f"   - Total pages scraped: {len(pages)}")
        print(f"   - Time taken: {duration:.2f} seconds")
        print(f"   - Average time per page: {duration/len(pages):.2f} seconds")
        
        # Show sample of scraped content
        if pages:
            print(f"\n📄 Sample page:")
            sample = pages[0]
            print(f"   - Title: {sample['title'][:100]}...")
            print(f"   - URL: {sample['url']}")
            print(f"   - Content length: {len(sample['content'])} characters")
        
    except KeyboardInterrupt:
        print("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
