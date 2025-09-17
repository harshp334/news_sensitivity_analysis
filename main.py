import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import sys
from datetime import datetime, timedelta
import argparse
from urllib.parse import urljoin, urlparse
import logging
import json

# Optional: you can set your API keys directly here (string).
# If left as None, the script will use --newsapi-key CLI arg or NEWSAPI_KEY env var.
# Load API keys from environment or a local `.env` file (do not rely on importing a dotfile as a module)
def _load_dotenv(path='.env'):
    data = {}
    try:
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"\' "')
                # strip surrounding quotes
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                data[k] = v
    except FileNotFoundError:
        pass
    return data

_DOTENV = _load_dotenv()
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY') or _DOTENV.get('NEWSAPI_KEY')
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_KEY') or _DOTENV.get('DEEPSEEK_KEY')

# Import DeepSeek enhancement module
try:
    from deepseek_enrichment import enrich_dataframe, save_enriched_data
except ImportError as e:
    print(f"Error importing deepseek_enrichment module: {e}")
    print("Make sure deepseek_enrichment.py is in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsAPIClient:
    """
    Integrated NewsAPI client for news data extraction
    """
    def __init__(self, api_key):
        """
        Initialize NewsAPI client
        
        Args:
            api_key (str): newsapi.org API key
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.headers = {"X-API-Key": self.api_key}
    
    def get_headlines(self, category=None, country="us", page_size=20, sources=None):
        """
        Get top headlines from NewsAPI
        
        Args:
            category (str): business, entertainment, general, health, science, sports, technology
            country (str): 2-letter country code (default: us)
            page_size (int): Number of articles to retrieve (max 100)
            sources (str): Comma-separated source IDs (overrides country/category)
        
        Returns:
            dict: API response with articles
        """
        endpoint = f"{self.base_url}/top-headlines"
        
        params = {
            "pageSize": page_size
        }
        
        if sources:
            params["sources"] = sources
        else:
            params["country"] = country
            if category:
                params["category"] = category
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching headlines: {e}")
            return None
    
    def search_articles(self, query, from_date=None, to_date=None, 
                       language="en", sort_by="publishedAt", page_size=20):
        """
        Search for articles using NewsAPI
        
        Args:
            query (str): Search keywords
            from_date (str): YYYY-MM-DD format
            to_date (str): YYYY-MM-DD format
            language (str): Language code (default: en)
            sort_by (str): relevancy, popularity, publishedAt
            page_size (int): Number of articles (max 100)
        
        Returns:
            dict: API response with articles
        """
        endpoint = f"{self.base_url}/everything"
        
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching articles: {e}")
            return None
    
    def get_sources(self, category=None, language="en", country="us"):
        """
        Get available news sources
        
        Args:
            category (str): Filter by category
            language (str): Language code
            country (str): Country code
        
        Returns:
            dict: Available sources
        """
        endpoint = f"{self.base_url}/sources"
        
        params = {
            "language": language,
            "country": country
        }
        
        if category:
            params["category"] = category
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching sources: {e}")
            return None

def articles_to_dataframe(api_response):
    """
    Convert NewsAPI response to pandas DataFrame
    
    Args:
        api_response (dict): NewsAPI response
    
    Returns:
        pd.DataFrame: Processed articles data
    """
    if not api_response or 'articles' not in api_response:
        return pd.DataFrame()
    
    articles = api_response['articles']
    
    # Extract relevant fields
    processed_articles = []
    for article in articles:
        processed_article = {
            'title': article.get('title'),
            'description': article.get('description'),
            'content': article.get('content'),
            'url': article.get('url'),
            'urlToImage': article.get('urlToImage'),
            'publishedAt': article.get('publishedAt'),
            'source_name': article.get('source', {}).get('name'),
            'source_id': article.get('source', {}).get('id'),
            'author': article.get('author')
        }
        processed_articles.append(processed_article)
    
    df = pd.DataFrame(processed_articles)
    
    # Convert publishedAt to datetime
    if not df.empty and 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['date'] = df['publishedAt'].dt.date
        df['hour'] = df['publishedAt'].dt.hour
    
    return df

def save_raw_data(df, filename="pipeline_raw_data.csv"):
    """
    Save raw data to CSV
    
    Args:
        df (pd.DataFrame): Articles dataframe
        filename (str): Output filename
    
    Returns:
        str: Filepath where data was saved
    """
    # Create data/raw directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    logger.info(f"Raw data saved to {filepath}")
    
    return filepath

class ReutersScraper:
    """
    Scraper for Reuters articles with rate limiting and error handling
    """
    
    def __init__(self, delay=2):
        """
        Initialize Reuters scraper
        
        Args:
            delay (int): Delay between requests in seconds
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_count = 0
    
    def is_reuters_url(self, url):
        """Check if URL is from Reuters"""
        if not url:
            return False
        return 'reuters.com' in url.lower()
    
    def scrape_article_content(self, url):
        """
        Scrape full article content from Reuters URL
        
        Args:
            url (str): Reuters article URL
            
        Returns:
            dict: Article content data
        """
        if not self.is_reuters_url(url):
            return {
                'scraped_content': '',
                'scraped_success': False,
                'scrape_error': 'Not a Reuters URL'
            }
        
        try:
            # Rate limiting
            if self.scraped_count > 0:
                time.sleep(self.delay)

            logger.info(f"Scraping: {url}")

            # Improve headers to reduce 401/403 blocks
            parsed = urlparse(url)
            self.session.headers.update({
                'Referer': f"{parsed.scheme}://{parsed.netloc}",
                'Accept-Language': 'en-US,en;q=0.9'
            })

            response = self.session.get(url, timeout=15)
            try:
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
            except requests.HTTPError as he:
                status = None
                if he.response is not None:
                    status = he.response.status_code
                # If blocked (401/403), attempt a fallback fetch via a text proxy
                if status in (401, 403):
                    try:
                        fallback_url = 'https://r.jina.ai/http://' + url.split('://', 1)[1]
                        logger.info(f"Primary fetch returned {status}; trying proxy fallback: {fallback_url}")
                        fb = self.session.get(fallback_url, timeout=15)
                        fb.raise_for_status()
                        soup = BeautifulSoup(fb.content, 'html.parser')
                    except Exception as fe:
                        logger.warning(f"Proxy fallback failed for {url}: {fe}")
                        raise
                else:
                    raise
            
            # Reuters article selectors (multiple fallbacks)
            content_selectors = [
                'div[data-testid="paragraph"]',  # Primary content
                '.StandardArticleBody_body',     # Alternative selector
                '[data-module="ArticleBody"] p', # Another alternative
                'div.ArticleBodyWrapper p',      # Fallback
                'article p'                      # Generic fallback
            ]
            
            content_paragraphs = []
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_paragraphs = [elem.get_text().strip() for elem in elements if elem.get_text().strip()]
                    break
            
            # Join paragraphs
            scraped_content = ' '.join(content_paragraphs)
            
            # Clean up content
            scraped_content = self._clean_content(scraped_content)
            
            self.scraped_count += 1
            
            return {
                'scraped_content': scraped_content,
                'scraped_success': True,
                'scrape_error': None,
                'scraped_timestamp': datetime.now().isoformat(),
                'content_length': len(scraped_content)
            }
            
        except requests.RequestException as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return {
                'scraped_content': '',
                'scraped_success': False,
                'scrape_error': str(e),
                'scraped_timestamp': datetime.now().isoformat(),
                'content_length': 0
            }
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return {
                'scraped_content': '',
                'scraped_success': False,
                'scrape_error': f"Parsing error: {str(e)}",
                'scraped_timestamp': datetime.now().isoformat(),
                'content_length': 0
            }
    
    def _clean_content(self, content):
        """Clean scraped content"""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = ' '.join(content.split())
        
        # Remove common Reuters footer text
        footer_phrases = [
            "Reporting by", "Additional reporting by", "Editing by",
            "Our Standards:", "Thomson Reuters Trust Principles"
        ]
        
        for phrase in footer_phrases:
            if phrase in content:
                content = content.split(phrase)[0]
        
        return content.strip()

class NewsAnalysisPipeline:
    """
    Main pipeline orchestrator with integrated NewsAPI functionality
    """
    
    def __init__(self, newsapi_key, deepseek_key):
        """
        Initialize the pipeline
        
        Args:
            newsapi_key (str): newsapi.org API key
            deepseek_key (str): deepseek.ai API key
        """
        self.newsapi_key = newsapi_key
        self.deepseek_key = deepseek_key
        self.newsapi_client = NewsAPIClient(newsapi_key)
        self.reuters_scraper = ReutersScraper()
        
        # Create directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/enriched", exist_ok=True)
        os.makedirs("examples", exist_ok=True)
        
        logger.info("Pipeline initialized successfully")
    
    def extract_news_data(self, categories=['technology', 'business'], 
                         search_terms=['Reuters'], max_articles=50, 
                         target_date=None):
        """
        Extract data from NewsAPI and identify Reuters articles
        
        Args:
            categories (list): NewsAPI categories to fetch
            search_terms (list): Search terms for finding articles
            max_articles (int): Maximum articles to process
            target_date (str): Target date in YYYY-MM-DD format (None for recent articles)
            
        Returns:
            pd.DataFrame: Raw news data
        """
        logger.info("Starting news data extraction...")
        
        # Handle date filtering
        if target_date:
            try:
                # Validate date format
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                # Create date range for the target date
                from_date = target_date
                to_date = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"Searching for articles from {target_date} to {to_date}")
                use_headlines = False  # Headlines endpoint doesn't support date filtering
            except ValueError:
                logger.error(f"Invalid date format: {target_date}. Use YYYY-MM-DD format.")
                return pd.DataFrame()
        else:
            # Default: get recent articles (last 7 days)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = None
            logger.info(f"Searching for recent articles from {from_date}")
            use_headlines = True  # Can use headlines for recent articles
        
        all_articles = []
        
        # Get headlines from categories (only if not filtering by specific date)
        if use_headlines:
            for category in categories:
                logger.info(f"Fetching recent {category} headlines...")
                response = self.newsapi_client.get_headlines(
                    category=category, 
                    page_size=min(20, max_articles // (len(categories) + len(search_terms)))
                )
                
                if response and 'articles' in response:
                    articles_df = articles_to_dataframe(response)
                    if not articles_df.empty:
                        all_articles.append(articles_df)
                        logger.info(f"Got {len(articles_df)} recent {category} articles")
                    else:
                        logger.warning(f"No articles found for category: {category}")
                else:
                    logger.warning(f"Failed to fetch {category} headlines")
        
        # Search for specific terms with date filtering
        for term in search_terms:
            if target_date:
                logger.info(f"Searching for '{term}' articles on {target_date}...")
            else:
                logger.info(f"Searching for recent '{term}' articles...")
            
            response = self.newsapi_client.search_articles(
                query=term,
                from_date=from_date,
                to_date=to_date,
                page_size=min(30, max_articles // len(search_terms))
            )
            
            if response and 'articles' in response:
                search_df = articles_to_dataframe(response)
                if not search_df.empty:
                    all_articles.append(search_df)
                    logger.info(f"Got {len(search_df)} '{term}' articles")
                else:
                    logger.warning(f"No articles found for search term: {term}")
            else:
                logger.warning(f"Failed to search for '{term}' articles")
        
        # If searching by specific date, also search categories using the search endpoint
        if target_date:
            for category in categories:
                logger.info(f"Searching for {category} articles on {target_date}...")
                
                # Use category-specific search terms
                category_terms = {
                    'technology': 'technology OR tech OR software OR AI',
                    'business': 'business OR finance OR economy OR market',
                    'health': 'health OR medical OR healthcare OR medicine',
                    'science': 'science OR research OR study OR discovery',
                    'sports': 'sports OR football OR basketball OR soccer',
                    'entertainment': 'entertainment OR movie OR music OR celebrity'
                }
                
                search_query = category_terms.get(category, category)
                
                response = self.newsapi_client.search_articles(
                    query=search_query,
                    from_date=from_date,
                    to_date=to_date,
                    page_size=min(15, max_articles // (len(categories) * 2))
                )
                
                if response and 'articles' in response:
                    category_df = articles_to_dataframe(response)
                    if not category_df.empty:
                        all_articles.append(category_df)
                        logger.info(f"Got {len(category_df)} {category} articles for {target_date}")
        
        # Combine all articles
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            
            # Remove duplicates based on URL
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['url'])
            combined_df = combined_df.head(max_articles)  # Limit total articles
            
            # Filter by exact date if specified (extra filtering for precision)
            if target_date:
                target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
                combined_df['article_date'] = pd.to_datetime(combined_df['publishedAt']).dt.date
                date_filtered_df = combined_df[combined_df['article_date'] == target_dt]
                
                if len(date_filtered_df) > 0:
                    combined_df = date_filtered_df
                    logger.info(f"Filtered to articles from exact date {target_date}: {len(combined_df)} articles")
                else:
                    logger.warning(f"No articles found for exact date {target_date}, keeping broader results")
            
            logger.info(f"Combined articles: {initial_count} → {len(combined_df)} (after deduplication)")
            logger.info(f"Total unique articles extracted: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No articles extracted from any source")
            return pd.DataFrame()
    
    def scrape_reuters_content(self, df, max_scrapes=20):
        """
        Scrape full content from Reuters articles
        
        Args:
            df (pd.DataFrame): DataFrame with article URLs
            max_scrapes (int): Maximum articles to scrape
            
        Returns:
            pd.DataFrame: DataFrame with scraped content
        """
        logger.info("Starting Reuters content scraping...")
        # Quick guard
        if df is None or df.empty:
            logger.warning("No articles provided to scrape")
            return df

        # Detect Reuters-origin articles. Some feeds use aggregator URLs (Yahoo, Memeorandum, etc.)
        url_mask = df['url'].str.contains('reuters.com', case=False, na=False)
        source_mask = df.get('source_name', pd.Series([''] * len(df))).str.contains('reuters', case=False, na=False)
        title_mask = df.get('title', pd.Series([''] * len(df))).str.contains(r'\bReuters\b', case=False, na=False)
        desc_mask = df.get('description', pd.Series([''] * len(df))).str.contains('Reuters', case=False, na=False)
        content_mask = df.get('content', pd.Series([''] * len(df))).str.contains('Reuters', case=False, na=False)

        reuters_origin_mask = url_mask | source_mask | title_mask | desc_mask | content_mask

        candidate_df = df[reuters_origin_mask].copy()
        logger.info(f"Found {len(candidate_df)} Reuters-origin candidate articles (limit: {max_scrapes})")

        if candidate_df.empty:
            logger.warning("No Reuters articles found for scraping")
            # Add empty scraping columns to original df
            out = df.copy()
            out['scraped_content'] = ''
            out['scraped_success'] = False
            out['scrape_error'] = 'No Reuters articles'
            out['scraped_timestamp'] = datetime.now().isoformat()
            out['content_length'] = 0
            return out

        # Limit how many we attempt to scrape
        candidate_df = candidate_df.head(max_scrapes)

        scraped_data = []

        for idx, row in candidate_df.iterrows():
            original_url = row.get('url')
            scrape_url = original_url

            # If the URL itself is not reuters.com, try to discover a canonical Reuters link on the page
            if original_url and 'reuters.com' not in original_url.lower():
                try:
                    resp = requests.get(original_url, timeout=10)
                    resp.raise_for_status()
                    page_soup = BeautifulSoup(resp.content, 'html.parser')

                    # Check for a canonical link first
                    canonical = page_soup.find('link', rel='canonical')
                    if canonical and canonical.get('href') and 'reuters.com' in canonical.get('href'):
                        scrape_url = canonical.get('href')
                    else:
                        # Fall back to scanning anchor tags for a Reuters URL
                        for a in page_soup.find_all('a', href=True):
                            href = a['href']
                            if 'reuters.com' in href.lower():
                                scrape_url = urljoin(original_url, href)
                                break
                except Exception as e:
                    logger.debug(f"Could not fetch aggregator page to find Reuters URL for {original_url}: {e}")

            logger.info(f"Scraping candidate article: {row.get('title','')[:60]} - source url: {original_url} -> scrape url: {scrape_url}")

            scrape_result = self.reuters_scraper.scrape_article_content(scrape_url)

            enhanced_row = row.to_dict()
            enhanced_row.update(scrape_result)
            scraped_data.append(enhanced_row)

        # Convert to DataFrame
        scraped_df = pd.DataFrame(scraped_data)

        # For articles we didn't attempt to scrape (non-candidates), add empty scraping columns
        non_candidate_df = df[~reuters_origin_mask].copy()
        if not non_candidate_df.empty:
            non_candidate_df['scraped_content'] = ''
            non_candidate_df['scraped_success'] = False
            non_candidate_df['scrape_error'] = 'Not a Reuters article'
            non_candidate_df['scraped_timestamp'] = datetime.now().isoformat()
            non_candidate_df['content_length'] = 0

        # Combine scraped and non-scraped articles
        if not non_candidate_df.empty:
            final_df = pd.concat([scraped_df, non_candidate_df], ignore_index=True)
        else:
            final_df = scraped_df

        logger.info(f"Successfully scraped {self.reuters_scraper.scraped_count} Reuters articles")
        return final_df
    
    def clean_data(self, df):
        """
        Clean and preprocess the data using pandas
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        initial_count = len(df)
        
        # Remove articles with no title or URL
        df = df.dropna(subset=['title', 'url'])
        logger.info(f"Removed articles without title/URL: {initial_count} → {len(df)}")
        
        # Clean text fields
        text_columns = ['title', 'description', 'content', 'scraped_content']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                # Remove extra whitespace
                df[col] = df[col].str.strip()
        
        # Remove articles with very short titles
        before_title_filter = len(df)
        df = df[df['title'].str.len() > 10]
        logger.info(f"Removed short titles: {before_title_filter} → {len(df)}")
        
        # Create combined content field for analysis
        df['analysis_text'] = df['title'] + '. ' + df['description'].fillna('')
        
        # Add scraped content if available and substantial
        def combine_content(row):
            base_text = row['analysis_text']
            if row.get('scraped_success', False) and len(row.get('scraped_content', '')) > 100:
                # Use first 500 chars of scraped content
                additional_content = row['scraped_content'][:500]
                return f"{base_text} {additional_content}"
            return base_text
        
        df['analysis_text'] = df.apply(combine_content, axis=1)
        
        # Remove duplicates by title similarity (basic)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['title'])
        logger.info(f"Removed duplicate titles: {before_dedup} → {len(df)}")
        
        # Filter out very short articles
        before_length_filter = len(df)
        df = df[df['analysis_text'].str.len() > 50]
        logger.info(f"Removed short articles: {before_length_filter} → {len(df)}")
        
        # Add data quality scores
        df['data_quality_score'] = self._calculate_quality_score(df)
        
        final_count = len(df)
        logger.info(f"Data cleaning complete: {initial_count} → {final_count} articles")
        
        return df
    
    def _calculate_quality_score(self, df):
        """Calculate a simple data quality score (0-10 scale)"""
        scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Title quality (0-3 points)
            title_len = len(str(row.get('title', '')))
            if title_len > 30:
                score += 2
            elif title_len > 15:
                score += 1
            
            # Description quality (0-2 points)
            desc_len = len(str(row.get('description', '')))
            if desc_len > 50:
                score += 2
            elif desc_len > 20:
                score += 1
            
            # Content availability (0-3 points)
            if row.get('scraped_success', False):
                score += 3
            elif len(str(row.get('content', ''))) > 100:
                score += 2
            elif len(str(row.get('content', ''))) > 50:
                score += 1
            
            # Source quality (0-2 points)
            source = str(row.get('source_name', '')).lower()
            quality_sources = ['reuters', 'bbc', 'cnn', 'ap', 'bloomberg']
            if any(qs in source for qs in quality_sources):
                score += 2
            
            scores.append(score)
        
        return scores
    
    def run_pipeline(self, max_articles=30, max_scrapes=15, 
                    categories=['technology', 'business'],
                    search_terms=['Reuters'], target_date=None):
        """
        Run the complete ETL pipeline
        
        Args:
            max_articles (int): Maximum articles to extract
            max_scrapes (int): Maximum articles to scrape
            categories (list): NewsAPI categories
            search_terms (list): Search terms
            target_date (str): Target date in YYYY-MM-DD format (None for recent)
            
        Returns:
            dict: Pipeline results
        """
        logger.info("="*60)
        logger.info("STARTING NEWS ANALYSIS PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        results = {}
        
        try:
            # STEP 1: EXTRACT
            logger.info("STEP 1: EXTRACTING NEWS DATA")
            if target_date:
                logger.info(f"Target: {max_articles} articles from {target_date}")
            else:
                logger.info(f"Target: {max_articles} recent articles from {len(categories)} categories + {len(search_terms)} search terms")
            
            raw_df = self.extract_news_data(categories, search_terms, max_articles, target_date)
            
            if raw_df.empty:
                error_msg = f"No data extracted for {'date ' + target_date if target_date else 'recent articles'}. Try a different date or broader search terms."
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
            
            results['extracted_count'] = len(raw_df)
            results['target_date'] = target_date
            
            # STEP 2: SCRAPE REUTERS CONTENT
            logger.info("STEP 2: SCRAPING REUTERS CONTENT")
            scraped_df = self.scrape_reuters_content(raw_df, max_scrapes)
            results['scraped_count'] = scraped_df['scraped_success'].sum()
            
            # STEP 3: CLEAN DATA
            logger.info("STEP 3: CLEANING DATA")
            clean_df = self.clean_data(scraped_df)
            results['cleaned_count'] = len(clean_df)
            
            if clean_df.empty:
                logger.error("No data remaining after cleaning. Pipeline terminated.")
                return {'success': False, 'error': 'No data after cleaning'}
            
            # Save raw data with date info in filename
            date_suffix = f"_{target_date}" if target_date else "_recent"
            raw_filename = f"pipeline_raw_data{date_suffix}.csv"
            raw_filepath = save_raw_data(clean_df, raw_filename)
            results['raw_data_path'] = raw_filepath
            
            # STEP 4: AI ENHANCEMENT
            logger.info("STEP 4: AI ENHANCEMENT WITH DEEPSEEK")
            logger.info(f"Enhancing {len(clean_df)} articles with sentiment, topic, and bias analysis...")
            enriched_df = enrich_dataframe(clean_df, self.deepseek_key, batch_size=5)
            results['enriched_count'] = len(enriched_df)
            
            # STEP 5: SAVE ENRICHED DATA
            logger.info("STEP 5: SAVING ENRICHED DATA")
            enriched_filename = f"pipeline_enriched_data{date_suffix}.csv"
            enriched_filepath = save_enriched_data(enriched_df, enriched_filename)
            results['enriched_data_path'] = enriched_filepath
            
            # STEP 6: CREATE EXAMPLES
            logger.info("STEP 6: CREATING EXAMPLES")
            self.create_examples(clean_df, enriched_df, target_date)
            
            # Final results
            end_time = datetime.now()
            results.update({
                'success': True,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': (end_time - start_time).total_seconds() / 60,
                'reuters_scraped': results['scraped_count'],
                'ai_enhanced': results['enriched_count']
            })
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Duration: {results['duration_minutes']:.1f} minutes")
            logger.info(f"Articles processed: {results['enriched_count']}")
            if target_date:
                logger.info(f"Date focus: {target_date}")
            logger.info(f"Reuters articles scraped: {results['scraped_count']}")
            logger.info(f"Data saved to:")
            logger.info(f"  - Raw: {results['raw_data_path']}")
            logger.info(f"  - Enriched: {results['enriched_data_path']}")
            logger.info("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def create_examples(self, raw_df, enriched_df, target_date=None):
        """Create before/after examples"""
        logger.info("Creating before/after examples...")
        
        # Select a few interesting articles for examples
        sample_articles = enriched_df.head(3)
        
        # Create markdown example
        date_info = f" - {target_date}" if target_date else " - Recent Articles"
        example_md = f"# News Analysis Pipeline - Before/After Examples{date_info}\n\n"
        example_md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if target_date:
            example_md += f"**Target Date**: {target_date}\n"
        example_md += "\n"
        example_md += f"**Pipeline Summary**:\n"
        example_md += f"- Total articles processed: {len(enriched_df)}\n"
        example_md += f"- Reuters articles scraped: {enriched_df['scraped_success'].sum()}\n"
        example_md += f"- Average data quality score: {enriched_df['data_quality_score'].mean():.1f}/10\n"
        
        if target_date:
            # Show date distribution
            enriched_df['article_date'] = pd.to_datetime(enriched_df['publishedAt']).dt.date
            date_counts = enriched_df['article_date'].value_counts().sort_index()
            example_md += f"- Articles by date:\n"
            for date, count in date_counts.head().items():
                example_md += f"  - {date}: {count} articles\n"
        
        example_md += "\n---\n\n"
        
        for i, (_, article) in enumerate(sample_articles.iterrows(), 1):
            example_md += f"## Example {i}: {article['title'][:60]}...\n\n"
            
            example_md += "### BEFORE (Raw Data):\n"
            example_md += f"- **Title**: {article['title']}\n"
            example_md += f"- **Source**: {article['source_name']}\n"
            example_md += f"- **Published**: {article.get('publishedAt', 'N/A')}\n"
            example_md += f"- **Description**: {article['description'][:150]}...\n"
            example_md += f"- **Data Quality Score**: {article.get('data_quality_score', 'N/A')}/10\n"
            
            if article.get('scraped_success', False):
                example_md += f"- **Scraped Content Length**: {article.get('content_length', 0)} characters\n"
            
            example_md += "\n"
            
            example_md += "### AFTER (AI Enhanced):\n"
            example_md += f"- **Sentiment**: {article.get('ai_sentiment', 'N/A')} ({article.get('ai_sentiment_confidence', 'N/A')} confidence)\n"
            example_md += f"- **Emotional Tone**: {article.get('ai_emotional_tone', 'N/A')}\n"
            example_md += f"- **Primary Category**: {article.get('ai_primary_category', 'N/A')}\n"
            example_md += f"- **Secondary Category**: {article.get('ai_secondary_category', 'N/A')}\n"
            example_md += f"- **Keywords**: {article.get('ai_keywords', 'N/A')}\n"
            example_md += f"- **Bias Level**: {article.get('ai_bias_level', 'N/A')}\n"
            example_md += f"- **Bias Direction**: {article.get('ai_bias_direction', 'N/A')}\n"
            example_md += f"- **Objectivity**: {article.get('ai_objectivity', 'N/A')}\n"
            
            reasoning = article.get('ai_sentiment_reasoning', 'N/A')
            if len(reasoning) > 5:  # Has actual reasoning
                example_md += f"- **AI Reasoning**: {reasoning}\n"
            
            example_md += "\n---\n\n"
        
        # Add summary statistics
        example_md += "## Summary Statistics\n\n"
        
        if 'ai_sentiment' in enriched_df.columns:
            sentiment_dist = enriched_df['ai_sentiment'].value_counts()
            example_md += "### Sentiment Distribution:\n"
            for sentiment, count in sentiment_dist.items():
                percentage = (count / len(enriched_df)) * 100
                example_md += f"- **{sentiment}**: {count} ({percentage:.1f}%)\n"
            example_md += "\n"
        
        if 'ai_primary_category' in enriched_df.columns:
            topic_dist = enriched_df['ai_primary_category'].value_counts()
            example_md += "### Topic Distribution:\n"
            for topic, count in topic_dist.items():
                percentage = (count / len(enriched_df)) * 100
                example_md += f"- **{topic}**: {count} ({percentage:.1f}%)\n"
            example_md += "\n"
        
        if 'ai_bias_level' in enriched_df.columns:
            bias_dist = enriched_df['ai_bias_level'].value_counts()
            example_md += "### Bias Level Distribution:\n"
            for bias, count in bias_dist.items():
                percentage = (count / len(enriched_df)) * 100
                example_md += f"- **{bias}**: {count} ({percentage:.1f}%)\n"
        
        # Save example with date suffix
        date_suffix = f"_{target_date}" if target_date else "_recent"
        example_filename = f"before_after_examples{date_suffix}.md"
        
        with open(f"examples/{example_filename}", 'w', encoding='utf-8') as f:
            f.write(example_md)
        
        # Save a sample CSV with top articles
        top_articles = enriched_df.nlargest(5, 'data_quality_score')
        sample_filename = f"sample_enriched_articles{date_suffix}.csv"
        top_articles.to_csv(f"examples/{sample_filename}", index=False)
        
        logger.info("Examples created successfully")

def test_newsapi_connection(api_key):
    """
    Test NewsAPI connection with a simple request
    
    Args:
        api_key (str): NewsAPI key to test
        
    Returns:
        bool: True if connection successful
    """
    logger.info("Testing NewsAPI connection...")
    
    client = NewsAPIClient(api_key)
    test_response = client.get_headlines(category="technology", page_size=5)
    
    if test_response and 'articles' in test_response:
        logger.info(f"✓ NewsAPI connection successful - got {len(test_response['articles'])} test articles")
        return True
    else:
        logger.error("✗ NewsAPI connection failed")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="News Analysis Pipeline - Complete ETL with AI Enhancement")
    
    # NewsAPI key (can be provided via CLI or NEWSAPI_KEY env var)
    parser.add_argument("--newsapi-key", required=False, help="NewsAPI key (or set NEWSAPI_KEY env var)")
    
    # Optional arguments
    parser.add_argument("--deepseek-key", required=False,
                       help="DeepSeek API key (default: provided key)")
    parser.add_argument("--max-articles", type=int, default=30, 
                       help="Maximum articles to extract (default: 30)")
    parser.add_argument("--max-scrapes", type=int, default=15, 
                       help="Maximum Reuters articles to scrape (default: 15)")
    parser.add_argument("--categories", nargs="+", default=["technology", "business"], 
                       help="NewsAPI categories (default: technology business)")
    parser.add_argument("--search-terms", nargs="+", default=["Reuters"], 
                       help="Search terms for articles (default: Reuters)")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test API connections and exit")
    
    args = parser.parse_args()
    
    # Resolve NewsAPI key: module-level `NEWSAPI_KEY` (if set) -> CLI arg -> NEWSAPI_KEY env var
    if NEWSAPI_KEY:
        newsapi_key = NEWSAPI_KEY
    else:
        newsapi_key = args.newsapi_key if args.newsapi_key else os.getenv('NEWSAPI_KEY')

    # If still missing, show a helpful error
    if not newsapi_key:
        parser.error("NewsAPI key is required: provide --newsapi-key or set NEWSAPI_KEY environment variable")

    # Test connection if requested
    if args.test_connection:
        logger.info("Testing API connections...")
        newsapi_ok = test_newsapi_connection(newsapi_key)

        # Also test DeepSeek if we resolved a key
        deepseek_ok = True
        if resolved_deepseek_key:
            try:
                from deepseek_enrichment import test_deepseek_connection
                deepseek_ok = test_deepseek_connection(resolved_deepseek_key)
            except Exception as e:
                logger.error(f"DeepSeek test helper not available: {e}")
                deepseek_ok = False

        if newsapi_ok and deepseek_ok:
            logger.info("✓ All API connections successful!")
            print("\n🎉 API connections are working!")
            print("You can now run the full pipeline without --test-connection")
        else:
            logger.error("✗ API connection test failed")
            if not newsapi_ok:
                print("\n❌ NewsAPI connection failed. Please check your NEWSAPI_KEY.")
            if not deepseek_ok:
                print("\n❌ DeepSeek connection failed. Please check your DEEPSEEK_KEY and permissions.")
            sys.exit(1)
        return
    
    # Display pipeline configuration
    logger.info("="*60)
    logger.info("NEWS ANALYSIS PIPELINE CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Max Articles: {args.max_articles}")
    logger.info(f"Max Reuters Scrapes: {args.max_scrapes}")
    logger.info(f"Categories: {', '.join(args.categories)}")
    logger.info(f"Search Terms: {', '.join(args.search_terms)}")
    # Log whether a DeepSeek key is configured without printing the key
    deepseek_key_present = bool(args.deepseek_key or NEWSAPI_KEY or DEEPSEEK_KEY)
    logger.info(f"DeepSeek API configured: {'Yes' if deepseek_key_present else 'No'}")
    logger.info("="*60)
    
    # Resolve DeepSeek key: prefer CLI arg -> module-level DEEPSEEK_KEY -> environment
    resolved_deepseek_key = args.deepseek_key or DEEPSEEK_KEY or os.getenv('DEEPSEEK_KEY')

    # Initialize and run pipeline
    try:
        pipeline = NewsAnalysisPipeline(newsapi_key, resolved_deepseek_key)

        results = pipeline.run_pipeline(
            max_articles=args.max_articles,
            max_scrapes=args.max_scrapes,
            categories=args.categories,
            search_terms=args.search_terms
        )
        
        if results['success']:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Processed {results['enriched_count']} articles")
            print(f"📰 Scraped {results['scraped_count']} Reuters articles")
            print(f"⏱️  Duration: {results['duration_minutes']:.1f} minutes")
            print(f"💾 Results saved to:")
            print(f"   - Raw: {results['raw_data_path']}")
            print(f"   - Enriched: {results['enriched_data_path']}")
            print(f"   - Examples: examples/before_after_examples.md")
            print(f"📝 Check pipeline.log for detailed logs")
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            print(f"📝 Check pipeline.log for error details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        print(f"📝 Check pipeline.log for full error details")
        sys.exit(1)

if __name__ == "__main__":
    main()