import time
from openai import AzureOpenAI
from src.config import config

class LLMClient:
    """
    A client for interacting with the Azure OpenAI service.
    """
    def __init__(self):
        if not all([config.AZURE_OPENAI_ENDPOINT, config.AZURE_OPENAI_API_KEY, config.LLM_DEPLOYMENT_NAME]):
            raise ValueError("Azure OpenAI environment variables are not set. Please check your .env file.")

        self.client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
        )
        self.deployment_name = config.LLM_DEPLOYMENT_NAME

    def generate(self, prompt: str, parameters: dict) -> tuple[str, float, dict, float]:
        """
        Generates a completion for a given prompt and returns the text, latency, token usage, and cost.
        """
        try:
            start_time = time.time()
            
            # Ensure 'messages' is not in parameters, as we set it
            parameters.pop("messages", None)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            
            latency = time.time() - start_time
            generated_text = response.choices[0].message.content.strip()
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            cost = (token_usage["prompt_tokens"] / 1000 * config.PROMPT_TOKEN_PRICE) + \
                   (token_usage["completion_tokens"] / 1000 * config.COMPLETION_TOKEN_PRICE)

            return generated_text, latency, token_usage, cost

        except Exception as e:
            print(f"An error occurred while generating text: {e}")
            return "Error: Could not generate response.", 0.0, {}, 0.0

llm_client = LLMClient() 