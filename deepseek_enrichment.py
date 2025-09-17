import requests
import json
import pandas as pd
import time
import os
from typing import Dict, List, Optional
import re
import logging

logger = logging.getLogger(__name__)
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
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                data[k] = v
    except FileNotFoundError:
        pass
    return data

_DOTENV = _load_dotenv()
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_KEY') or _DOTENV.get('DEEPSEEK_KEY')

class DeepSeekEnhancer:
    def __init__(self, api_key):
        """
        Initialize DeepSeek API client for news analysis
        
        Args:
            api_key (str): DeepSeek API key
        """
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.request_count = 0
        self.rate_limit_delay = 1  # seconds between requests
        # Validate API key early to surface helpful error messages
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_KEY in environment or pass via arguments.")
    
    def _make_api_request(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """
        Make a request to DeepSeek API with rate limiting
        
        Args:
            prompt (str): The prompt to send
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: API response content or None if failed
        """
        # Rate limiting
        if self.request_count > 0:
            time.sleep(self.rate_limit_delay)
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for consistent analysis
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            return result['choices'][0]['message']['content'].strip()
        
        except requests.exceptions.HTTPError as he:
            status = he.response.status_code if he.response is not None else 'unknown'
            # Provide clearer guidance for 401/403
            if status == 401:
                logger.error("DeepSeek API returned 401 Unauthorized - check your DEEPSEEK_KEY and permissions")
            elif status == 403:
                logger.error("DeepSeek API returned 403 Forbidden - access may be restricted for this key")
            else:
                logger.error(f"DeepSeek HTTP error: {status} - {he}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        """
        Analyze sentiment of news article text
        
        Args:
            text (str): Article title + description or full content
            
        Returns:
            dict: Sentiment analysis results
        """
        prompt = f"""
        Analyze the sentiment of this news article text. Provide your analysis in this EXACT format:

        SENTIMENT: [Positive/Negative/Neutral]
        CONFIDENCE: [High/Medium/Low]
        EMOTIONAL_TONE: [one word describing the dominant emotion]
        REASONING: [brief explanation in 10 words or less]

        Text to analyze:
        {text[:500]}...
        """
        
        response = self._make_api_request(prompt, max_tokens=100)
        
        if not response:
            return {
                'sentiment': 'Unknown',
                'confidence': 'Low',
                'emotional_tone': 'Neutral',
                'reasoning': 'API call failed'
            }
        
        # Parse structured response
        sentiment_data = {
            'sentiment': 'Neutral',
            'confidence': 'Medium',
            'emotional_tone': 'Neutral',
            'reasoning': 'No clear sentiment detected'
        }
        
        lines = response.split('\n')
        for line in lines:
            if 'SENTIMENT:' in line:
                sentiment_data['sentiment'] = line.split(':', 1)[1].strip()
            elif 'CONFIDENCE:' in line:
                sentiment_data['confidence'] = line.split(':', 1)[1].strip()
            elif 'EMOTIONAL_TONE:' in line:
                sentiment_data['emotional_tone'] = line.split(':', 1)[1].strip()
            elif 'REASONING:' in line:
                sentiment_data['reasoning'] = line.split(':', 1)[1].strip()
        
        return sentiment_data
    
    def categorize_topic(self, text: str) -> Dict[str, str]:
        """
        Categorize news article by topic
        
        Args:
            text (str): Article title + description
            
        Returns:
            dict: Topic categorization results
        """
        prompt = f"""
        Categorize this news article into topics. Use this EXACT format:

        PRIMARY_CATEGORY: [Politics/Business/Technology/Health/Sports/Entertainment/Science/World/Crime/Environment]
        SECONDARY_CATEGORY: [same options as above, or "None" if not applicable]
        KEYWORDS: [3-5 relevant keywords separated by commas]
        SPECIFICITY: [General/Specific/Highly_Specific]

        Text to categorize:
        {text[:400]}...
        """
        
        response = self._make_api_request(prompt, max_tokens=120)
        
        if not response:
            return {
                'primary_category': 'General',
                'secondary_category': 'None',
                'keywords': 'news',
                'specificity': 'General'
            }
        
        # Parse structured response
        topic_data = {
            'primary_category': 'General',
            'secondary_category': 'None',
            'keywords': 'news',
            'specificity': 'General'
        }
        
        lines = response.split('\n')
        for line in lines:
            if 'PRIMARY_CATEGORY:' in line:
                topic_data['primary_category'] = line.split(':', 1)[1].strip()
            elif 'SECONDARY_CATEGORY:' in line:
                topic_data['secondary_category'] = line.split(':', 1)[1].strip()
            elif 'KEYWORDS:' in line:
                topic_data['keywords'] = line.split(':', 1)[1].strip()
            elif 'SPECIFICITY:' in line:
                topic_data['specificity'] = line.split(':', 1)[1].strip()
        
        return topic_data
    
    def detect_bias(self, text: str, source_name: str = "") -> Dict[str, str]:
        """
        Detect potential editorial bias in news article
        
        Args:
            text (str): Article content
            source_name (str): Name of news source
            
        Returns:
            dict: Bias detection results
        """
        prompt = f"""
        Analyze this news text for potential editorial bias. Use this EXACT format:

        BIAS_LEVEL: [Low/Medium/High]
        BIAS_DIRECTION: [Liberal/Conservative/Neutral/Commercial/Sensationalist]
        OBJECTIVITY: [Objective/Somewhat_Objective/Opinion-Heavy/Clearly_Biased]
        INDICATORS: [brief description of bias indicators, max 15 words]

        Source: {source_name}
        Text to analyze:
        {text[:600]}...
        """
        
        response = self._make_api_request(prompt, max_tokens=120)
        
        if not response:
            return {
                'bias_level': 'Unknown',
                'bias_direction': 'Neutral',
                'objectivity': 'Unknown',
                'indicators': 'Analysis failed'
            }
        
        # Parse structured response
        bias_data = {
            'bias_level': 'Low',
            'bias_direction': 'Neutral',
            'objectivity': 'Objective',
            'indicators': 'Standard reporting language'
        }
        
        lines = response.split('\n')
        for line in lines:
            if 'BIAS_LEVEL:' in line:
                bias_data['bias_level'] = line.split(':', 1)[1].strip()
            elif 'BIAS_DIRECTION:' in line:
                bias_data['bias_direction'] = line.split(':', 1)[1].strip()
            elif 'OBJECTIVITY:' in line:
                bias_data['objectivity'] = line.split(':', 1)[1].strip()
            elif 'INDICATORS:' in line:
                bias_data['indicators'] = line.split(':', 1)[1].strip()
        
        return bias_data
    
    def comprehensive_analysis(self, title: str, description: str, 
                             content: str = "", source_name: str = "") -> Dict:
        """
        Perform comprehensive analysis combining all three methods
        
        Args:
            title (str): Article title
            description (str): Article description
            content (str): Full article content (optional)
            source_name (str): Source name
            
        Returns:
            dict: Complete analysis results
        """
        # Combine text for analysis
        analysis_text = f"{title}. {description}"
        if content:
            analysis_text += f" {content[:300]}"  # Limit content length
        
        print(f"Analyzing: {title[:50]}...")
        
        # Perform all three analyses
        sentiment_results = self.analyze_sentiment(analysis_text)
        topic_results = self.categorize_topic(analysis_text)
        bias_results = self.detect_bias(analysis_text, source_name)
        
        # Combine results
        comprehensive_results = {
            # Original data
            'title': title,
            'description': description,
            'source_name': source_name,
            
            # Sentiment Analysis
            'ai_sentiment': sentiment_results['sentiment'],
            'ai_sentiment_confidence': sentiment_results['confidence'],
            'ai_emotional_tone': sentiment_results['emotional_tone'],
            'ai_sentiment_reasoning': sentiment_results['reasoning'],
            
            # Topic Categorization
            'ai_primary_category': topic_results['primary_category'],
            'ai_secondary_category': topic_results['secondary_category'],
            'ai_keywords': topic_results['keywords'],
            'ai_specificity': topic_results['specificity'],
            
            # Bias Detection
            'ai_bias_level': bias_results['bias_level'],
            'ai_bias_direction': bias_results['bias_direction'],
            'ai_objectivity': bias_results['objectivity'],
            'ai_bias_indicators': bias_results['indicators'],
            
            # Metadata
            'ai_analysis_timestamp': pd.Timestamp.now().isoformat(),
            'ai_total_requests': 3  # One for each analysis type
        }
        
        return comprehensive_results

def enrich_dataframe(df: pd.DataFrame, deepseek_api_key: str, 
                    batch_size: int = 10) -> pd.DataFrame:
    """
    Enrich entire DataFrame with DeepSeek analysis
    
    Args:
        df (pd.DataFrame): DataFrame with news articles
        deepseek_api_key (str): DeepSeek API key
        batch_size (int): Number of articles to process at once
        
    Returns:
        pd.DataFrame: Enhanced DataFrame with AI analysis
    """
    enhancer = DeepSeekEnhancer(deepseek_api_key)
    enriched_data = []
    
    print(f"Starting DeepSeek enhancement of {len(df)} articles...")
    print(f"Processing in batches of {batch_size}")
    
    for i, row in df.iterrows():
        try:
            # Get comprehensive analysis
            analysis = enhancer.comprehensive_analysis(
                title=row.get('title', ''),
                description=row.get('description', ''),
                content=row.get('content', ''),
                source_name=row.get('source_name', '')
            )
            
            # Combine original data with analysis
            enriched_row = {**row.to_dict(), **analysis}
            enriched_data.append(enriched_row)
            
            # Progress update
            if (i + 1) % batch_size == 0:
                print(f"Processed {i + 1}/{len(df)} articles...")
                
        except Exception as e:
            print(f"Error processing article {i}: {e}")
            # Add original data with error markers
            error_row = row.to_dict()
            error_row.update({
                'ai_sentiment': 'Error',
                'ai_primary_category': 'Error',
                'ai_bias_level': 'Error',
                'ai_error': str(e)
            })
            enriched_data.append(error_row)
    
    print(f"Enhancement complete! Processed {len(enriched_data)} articles")
    print(f"Total API requests made: {enhancer.request_count}")
    
    return pd.DataFrame(enriched_data)

def save_enriched_data(df: pd.DataFrame, filename: str = "enriched_news_analysis.csv"):
    """
    Save enriched data to CSV file
    
    Args:
        df (pd.DataFrame): Enriched DataFrame
        filename (str): Output filename
    """
    # Create data/enriched directory if it doesn't exist
    os.makedirs("data/enriched", exist_ok=True)
    
    filepath = f"data/enriched/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Enriched data saved to {filepath}")
    
    # Save a summary report
    summary_filepath = f"data/enriched/analysis_summary.txt"
    with open(summary_filepath, 'w') as f:
        f.write("DeepSeek News Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Articles Analyzed: {len(df)}\n")
        f.write(f"Analysis Timestamp: {pd.Timestamp.now()}\n\n")
        
        # Sentiment distribution
        if 'ai_sentiment' in df.columns:
            sentiment_dist = df['ai_sentiment'].value_counts()
            f.write("Sentiment Distribution:\n")
            for sentiment, count in sentiment_dist.items():
                f.write(f"  {sentiment}: {count}\n")
        
        # Topic distribution  
        if 'ai_primary_category' in df.columns:
            topic_dist = df['ai_primary_category'].value_counts()
            f.write("\nTopic Distribution:\n")
            for topic, count in topic_dist.items():
                f.write(f"  {topic}: {count}\n")
        
        # Bias distribution
        if 'ai_bias_level' in df.columns:
            bias_dist = df['ai_bias_level'].value_counts()
            f.write("\nBias Level Distribution:\n")
            for bias, count in bias_dist.items():
                f.write(f"  {bias}: {count}\n")
    
    print(f"Summary report saved to {summary_filepath}")
    return filepath

# Testing function
def test_deepseek_enhancement():
    """
    Test the DeepSeek enhancement with sample data
    """
    # Sample news data for testing
    sample_data = {
        'title': ['Tech Giants Report Strong Q3 Earnings', 
                 'Climate Change Protests Spread Across Europe',
                 'New Medical Breakthrough Offers Hope for Cancer Patients'],
        'description': ['Major technology companies exceeded Wall Street expectations with robust quarterly results.',
                       'Environmental activists demand immediate action on climate policies in major European cities.',
                       'Researchers announce promising results from innovative cancer treatment trials.'],
        'source_name': ['Reuters', 'CNN', 'BBC'],
        'url': ['https://example.com/1', 'https://example.com/2', 'https://example.com/3']
    }
    
    test_df = pd.DataFrame(sample_data)
    
    print("Testing DeepSeek Enhancement...")
    print("-" * 50)
    
    # Load API key from .env or environment for testing
    api_key = DEEPSEEK_KEY or os.environ.get('DEEPSEEK_KEY')
    if not api_key:
        print("No DeepSeek API key found in .env or environment. Set DEEPSEEK_KEY to run tests.")
        return pd.DataFrame()
    
    # Enrich the sample data
    enriched_df = enrich_dataframe(test_df, api_key, batch_size=3)
    
    # Display results
    print("\nEnhancement Results:")
    print("=" * 50)
    
    for i, row in enriched_df.iterrows():
        print(f"\nArticle {i+1}: {row['title']}")
        print(f"  Sentiment: {row.get('ai_sentiment', 'N/A')} ({row.get('ai_sentiment_confidence', 'N/A')} confidence)")
        print(f"  Category: {row.get('ai_primary_category', 'N/A')}")
        print(f"  Keywords: {row.get('ai_keywords', 'N/A')}")
        print(f"  Bias Level: {row.get('ai_bias_level', 'N/A')}")
        print(f"  Objectivity: {row.get('ai_objectivity', 'N/A')}")
    
    # Save test results
    save_enriched_data(enriched_df, "test_enrichment_results.csv")
    
    return enriched_df


def test_deepseek_connection(api_key: str) -> bool:
    """
    Test DeepSeek connection with a minimal request. Returns True if authenticated.
    """
    try:
        enhancer = DeepSeekEnhancer(api_key)
        # Use a minimal prompt to validate authentication and basic response
        response = enhancer._make_api_request("Say hi", max_tokens=5)
        if response is None:
            logger.error("DeepSeek connection test failed: no response (possible auth error)")
            return False
        return True
    except ValueError as ve:
        logger.error(f"DeepSeek connection test failed: {ve}")
        return False
    except Exception as e:
        logger.error(f"DeepSeek connection test unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    test_deepseek_enhancement()