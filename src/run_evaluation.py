import os
import glob
import pandas as pd
import yaml
import time
import tempfile
from src.llm_client import llm_client
from src.config import config
from src.prompt_config import get_params_for_prompt

# Updated to use the azure-ai-evaluation SDK
from azure.ai.evaluation import evaluate, CoherenceEvaluator, FluencyEvaluator, RelevanceEvaluator

# Custom Evaluator example
def keyword_match_evaluator(ground_truth_keywords, answer, **kwargs):
    """
    Custom evaluator to check for keyword presence.
    """
    if not ground_truth_keywords or not isinstance(ground_truth_keywords, str):
        return {"score": 0.0, "reason": "Ground truth keywords are empty or invalid."}

    keywords = [k.strip().lower() for k in ground_truth_keywords.split(',')]
    present_keywords = [k for k in keywords if k in answer.lower()]
    score = len(present_keywords) / len(keywords) if keywords else 0
    
    return {"keyword_match_score": score, "present_keywords": present_keywords}


def load_prompts():
    """Loads all prompts from the prompt directory."""
    prompts = []
    prompt_files = glob.glob(os.path.join(config.PROMPT_DIR, "*.yaml"))
    for file_path in prompt_files:
        with open(file_path, 'r') as f:
            prompts.append(yaml.safe_load(f))
    return prompts

def generate_responses(prompts, topics_df):
    """Generates LLM responses for each prompt and topic."""
    records = []
    for prompt_info in prompts:
        for _, row in topics_df.iterrows():
            topic = row['topic']
            
            prompt_template = prompt_info['template']
            prompt_params = get_params_for_prompt(prompt_info['name'])
            
            prompt = prompt_template.format(topic=topic)
            
            generated_text, latency, token_usage, cost = llm_client.generate(prompt, prompt_params)
            
            record = {
                "prompt_name": prompt_info['name'],
                "topic": topic,
                "generated_text": generated_text,
                "ground_truth_keywords": row['ground_truth_keywords'],
                "latency": latency,
                "cost": cost,
                **{f"param_{k}": v for k, v in prompt_params.items()},
                **token_usage,
            }
            records.append(record)
            
            print(f"Generated response for prompt '{prompt_info['name']}' on topic '{topic}'")
    return pd.DataFrame(records)

def run_pf_evaluation(data_df):
    """Runs evaluations using the Azure AI SDK evaluators."""

    print("Running evaluations with Azure AI SDK (Coherence, Fluency, Relevance)...")
    
    # Create a temporary file to hold the data for the evaluation SDK
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl", encoding="utf-8") as f:
        data_df.to_json(f, orient="records", lines=True)
        temp_data_path = f.name

    model_config = {
        "api_version": config.OPENAI_API_VERSION,
        "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "azure_deployment": config.LLM_DEPLOYMENT_NAME,
        "api_key": config.AZURE_OPENAI_API_KEY,
    }

    coherence_evaluator = CoherenceEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    relevance_evaluator = RelevanceEvaluator(model_config)

    try:
        eval_result = evaluate(
            data=temp_data_path,
            evaluators={
                "coherence": coherence_evaluator,
                "fluency": fluency_evaluator,
                "relevance": relevance_evaluator,
            },
            # This is the key change: we map our dataframe columns (e.g., 'topic')
            # to the names the evaluators expect (e.g., 'question').
            # This keeps our code clean and intuitive.
            evaluator_config={
                "coherence": {
                    "question": "${data.topic}",
                    "answer": "${data.generated_text}",
                },
                "fluency": {
                    "question": "${data.topic}",
                    "answer": "${data.generated_text}",
                },
                "relevance": {
                    "question": "${data.topic}",
                    "answer": "${data.generated_text}",
                    "context": "${data.ground_truth_keywords}",
                },
            },
        )
        
        if "metrics_summary" not in eval_result or "results" not in eval_result:
            print("Evaluation failed. Full result:")
            import json
            # Pretty print the dict
            print(json.dumps(eval_result, indent=2))
            return pd.DataFrame(), pd.DataFrame()

        # The result object contains both summary metrics and per-row data
        summary_df = pd.DataFrame.from_dict(eval_result["metrics_summary"], orient='index').transpose()
        results_df = pd.DataFrame(eval_result["results"])
        
        return summary_df, results_df
    finally:
        os.remove(temp_data_path)


def flatten_and_clean_results(df):
    """
    Transforms the raw, nested DataFrame from the evaluation result into a clean,
    flat table suitable for analysis.
    """
    # Create a new DataFrame to hold the cleaned data
    clean_df = pd.DataFrame()

    # The raw dataframe has columns like 'inputs.topic', 'outputs.coherence.score', etc.
    # We will extract the parts we need.
    
    # Extract data from the 'inputs' part of the raw output
    for col in ['prompt_name', 'topic', 'generated_text', 'ground_truth_keywords', 
                'latency', 'cost', 'prompt_tokens', 'completion_tokens', 'total_tokens']:
        if f'inputs.{col}' in df.columns:
            clean_df[col] = df[f'inputs.{col}']

    # Extract model parameters (which start with 'param_')
    param_cols = [c for c in df.columns if c.startswith('inputs.param_')]
    for col in param_cols:
        clean_name = col.replace('inputs.', '')
        clean_df[clean_name] = df[col]

    # Extract evaluator scores from the 'outputs' part
    if 'outputs.coherence.score' in df.columns:
        clean_df['coherence'] = df['outputs.coherence.score']
    if 'outputs.fluency.score' in df.columns:
        clean_df['fluency'] = df['outputs.fluency.score']
    if 'outputs.relevance.score' in df.columns:
        clean_df['relevance'] = df['outputs.relevance.score']
        
    # Add the custom evaluator results (they are not nested)
    if 'keyword_match_score' in df.columns:
        clean_df['keyword_match_score'] = df['keyword_match_score']

    # Define the desired column order for the final report
    final_column_order = [
        'prompt_name', 'topic', 'coherence', 'fluency', 'relevance', 'keyword_match_score',
        'latency', 'cost', 'prompt_tokens', 'completion_tokens', 'total_tokens'
    ]
    # Add any parameter columns to the order
    param_clean_cols = [c.replace('inputs.', '') for c in param_cols]
    final_column_order.extend(param_clean_cols)
    final_column_order.append('generated_text') # Keep the text at the end

    # Reorder the dataframe, only keeping the columns that actually exist
    existing_cols = [c for c in final_column_order if c in clean_df.columns]
    
    return clean_df[existing_cols]


def main():
    """Main function to run the evaluation pipeline."""
    if not all([config.AZURE_OPENAI_API_KEY, config.AZURE_OPENAI_ENDPOINT, config.LLM_DEPLOYMENT_NAME]):
        print("Azure OpenAI credentials are not set. Please create a .env file and set the required variables.")
        return

    prompts = load_prompts()
    topics_df = pd.read_csv(config.EVAL_TOPICS_FILE)
    responses_df = generate_responses(prompts, topics_df)
    
    eval_summary_df, pf_results_df = run_pf_evaluation(responses_df)
    
    if pf_results_df.empty:
        print("Evaluation using Azure AI SDK failed. Skipping final report generation.")
        # Still save the generated responses
        os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
        responses_df.to_csv(config.OUTPUT_FILE, index=False)
        print(f"\nGeneration results saved to {config.OUTPUT_FILE}, but evaluation failed.")
        return

    print("Running custom evaluations (Keyword Match)...")
    # The promptflow evaluators return the input data plus the results.
    # We can run the custom evaluator on this combined dataframe.
    custom_eval_results = pf_results_df.apply(
        lambda row: keyword_match_evaluator(row['ground_truth_keywords'], row['generated_text']), 
        axis=1, 
        result_type='expand'
    )

    # Combine the promptflow results (which includes original data) and custom results
    raw_final_df = pd.concat([
        pf_results_df.reset_index(drop=True),
        custom_eval_results.reset_index(drop=True)
    ], axis=1)
    
    print("\n--- Cleaning and structuring final results ---")
    final_df = flatten_and_clean_results(raw_final_df)

    os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(config.OUTPUT_FILE, index=False)
    
    print(f"\nEvaluation complete. Results saved to {config.OUTPUT_FILE}")
    print("\n--- Evaluation Metrics Summary ---")
    print(eval_summary_df.to_string())
    print("\n--- Sample of Full Per-Row Results ---")
    print(final_df.head().to_string())


if __name__ == "__main__":
    main() 