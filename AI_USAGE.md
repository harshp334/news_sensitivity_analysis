# Documentation of AI Usage

## AI Prompts Given

1. Get headlines from NewsAPI (I will enter the API key myself). Extract Reuters URLs from headlines. Scrape full article content from Reuters. Clean and merge datasets with pandas. Send to DeepSeek for AI enhancement (I will enter the API key myself). Enhance for sentiment analysis, topic categorization, and bias detection. Save raw and enriched data.

2. Create the corresponding DeepSeek enhancement module.

3. Instead of a separate news integration file, please incorporate directly into main.py.

4. Apply a fallback that allows me to enter the NewsAPI key as an env var.

5. The script is not scraping any Reuters articles. Please update detection and scraping logic.

6. Allow me to input my NewsAPI key directly into the script.

## Human-Written Code

- "Combine results" section of deepseek_enrichment

- Summary report section of deepseek_enrichment

- Article search section of main