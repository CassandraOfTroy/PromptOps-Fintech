import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """
    Configuration class for the application.
    Loads environment variables and sets up constants.
    """
    # Azure OpenAI credentials
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Azure AI Foundry Configuration
    AZURE_AI_FOUNDRY_WORKSPACE = os.getenv("AZURE_AI_FOUNDRY_WORKSPACE")
    AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")

    # LLM Deployment
    LLM_DEPLOYMENT_NAME = os.getenv("LLM_DEPLOYMENT_NAME")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")


    # Project paths
    PROMPT_DIR = "prompts/investment_blog"
    EVAL_TOPICS_FILE = "data/evaluation_topics.csv"
    OUTPUT_FILE = "outputs/evaluation_results.csv"

    # Pricing for cost calculation (example prices, update as needed)
    # Prices per 1K tokens for a model like gpt-3.5-turbo
    PROMPT_TOKEN_PRICE = 0.0015
    COMPLETION_TOKEN_PRICE = 0.002
    
    # --- IMPORTANT:Define minimum acceptable scores for evaluation metrics ---
    # These values are examples. Adjust them to your quality standards.
    EVALUATION_THRESHOLDS = {
        "coherence": 3.5,
        "fluency": 3.5,
        "relevance": 4.0,
        "keyword_match_score": 0.2
    }

config = Config()