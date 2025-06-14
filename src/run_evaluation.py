import os
import glob
import pandas as pd
import yaml
import time
import tempfile
import sys # Import the sys module to exit the script
from datetime import datetime
from src.llm_client import llm_client
from src.config import config
from src.prompt_config import get_params_for_prompt

# Use the azure-ai-evaluation SDK
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
    
    model_config = {
        "api_version": config.OPENAI_API_VERSION,
        "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "azure_deployment": config.LLM_DEPLOYMENT_NAME,
        "api_key": config.AZURE_OPENAI_API_KEY,
    }

    coherence_evaluator = CoherenceEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    relevance_evaluator = RelevanceEvaluator(model_config)

    results = []
    
    for idx, row in data_df.iterrows():
        try:
            question = row['topic']
            answer = row['generated_text']
            context = row['ground_truth_keywords']
            
            print(f"Evaluating row {idx + 1}/{len(data_df)}: {row['prompt_name']} - {question[:50]}...")
            
            coherence_result = coherence_evaluator(query=question, response=answer)
            fluency_result = fluency_evaluator(response=answer)
            relevance_result = relevance_evaluator(query=question, response=answer)
            
            result_row = {
                'inputs.prompt_name': row['prompt_name'],
                'inputs.question': question,
                'inputs.answer': answer,
                'inputs.context': context,
                'inputs.latency': row['latency'],
                'inputs.cost': row['cost'],
                'inputs.prompt_tokens': row['prompt_tokens'],
                'inputs.completion_tokens': row['completion_tokens'],
                'inputs.total_tokens': row['total_tokens'],
                'outputs.coherence.score': coherence_result.get('coherence', coherence_result.get('score', 0)),
                'outputs.fluency.score': fluency_result.get('fluency', fluency_result.get('score', 0)),
                'outputs.relevance.score': relevance_result.get('relevance', relevance_result.get('score', 0)),
            }
            
            for col in data_df.columns:
                if col.startswith('param_'):
                    result_row[f'inputs.{col}'] = row[col]
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error evaluating row {idx}: {e}")
            result_row = {
                'inputs.prompt_name': row['prompt_name'],
                'inputs.question': row['topic'],
                'inputs.answer': row['generated_text'],
                'inputs.context': row['ground_truth_keywords'],
                'inputs.latency': row['latency'],
                'inputs.cost': row['cost'],
                'inputs.prompt_tokens': row['prompt_tokens'],
                'inputs.completion_tokens': row['completion_tokens'],
                'inputs.total_tokens': row['total_tokens'],
                'outputs.coherence.score': None,
                'outputs.fluency.score': None,
                'outputs.relevance.score': None,
            }
            
            for col in data_df.columns:
                if col.startswith('param_'):
                    result_row[f'inputs.{col}'] = row[col]
            
            results.append(result_row)
    
    if not results:
        print("No evaluation results generated.")
        return pd.DataFrame(), pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    summary_data = {}
    for metric in ['coherence', 'fluency', 'relevance']:
        scores = results_df[f'outputs.{metric}.score'].dropna()
        if len(scores) > 0:
            summary_data[f'{metric}.mean'] = scores.mean()
            summary_data[f'{metric}.std'] = scores.std()
        else:
            summary_data[f'{metric}.mean'] = None
            summary_data[f'{metric}.std'] = None
    
    summary_df = pd.DataFrame([summary_data])
    
    return summary_df, results_df


def flatten_and_clean_results(df):
    """
    Transforms the raw, nested DataFrame from the evaluation result into a clean,
    flat table suitable for analysis.
    """
    clean_df = pd.DataFrame()
    
    if 'inputs.question' in df.columns:
        clean_df['topic'] = df['inputs.question']
    if 'inputs.answer' in df.columns:
        clean_df['generated_text'] = df['inputs.answer']
    
    passthrough_cols = ['prompt_name', 'latency', 'cost', 'prompt_tokens', 'completion_tokens', 'total_tokens']
    for col in passthrough_cols:
        if f'inputs.{col}' in df.columns:
            clean_df[col] = df[f'inputs.{col}']

    param_cols = [c for c in df.columns if c.startswith('inputs.param_')]
    for col in param_cols:
        clean_name = col.replace('inputs.', '')
        clean_df[clean_name] = df[col]

    if 'outputs.coherence.score' in df.columns:
        clean_df['coherence'] = df['outputs.coherence.score']
    if 'outputs.fluency.score' in df.columns:
        clean_df['fluency'] = df['outputs.fluency.score']
    if 'outputs.relevance.score' in df.columns:
        clean_df['relevance'] = df['outputs.relevance.score']
        
    if 'keyword_match_score' in df.columns:
        clean_df['keyword_match_score'] = df['keyword_match_score']

    final_column_order = [
        'prompt_name', 'topic', 'coherence', 'fluency', 'relevance', 'keyword_match_score',
        'latency', 'cost', 'prompt_tokens', 'completion_tokens', 'total_tokens'
    ]
    param_clean_cols = [c.replace('inputs.', '') for c in param_cols]
    final_column_order.extend(param_clean_cols)
    final_column_order.append('generated_text')

    existing_cols = [c for c in final_column_order if c in clean_df.columns]
    
    return clean_df[existing_cols]

# --- Function to validate metrics against thresholds ---
def validate_metrics(metrics_df, thresholds):
    """
    Checks if any metric in the DataFrame is below its defined threshold.
    Returns a list of failure messages.
    """
    print("\n--- Validating metrics against thresholds ---")
    failures = []
    
    # Check each individual result row against the thresholds
    for index, row in metrics_df.iterrows():
        for metric, threshold in thresholds.items():
            # Check if the metric exists in the row and is not a NaN value
            if metric in row and pd.notna(row[metric]):
                if row[metric] < threshold:
                    failure_message = (
                        f"VALIDATION FAILED for prompt '{row['prompt_name']}' on topic '{row['topic']}': "
                        f"Metric '{metric}' score ({row[metric]:.2f}) is below threshold ({threshold})."
                    )
                    print(failure_message)
                    failures.append(failure_message)
                    
    return failures

def main():
    """Main function to run the evaluation pipeline."""
    if not all([config.AZURE_OPENAI_API_KEY, config.AZURE_OPENAI_ENDPOINT, config.LLM_DEPLOYMENT_NAME]):
        print("Azure OpenAI credentials are not set. Please create a .env file and set the required variables.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    prompts = load_prompts()
    topics_df = pd.read_csv(config.EVAL_TOPICS_FILE)
    responses_df = generate_responses(prompts, topics_df)
    
    eval_summary_df, pf_results_df = run_pf_evaluation(responses_df)
    
    if pf_results_df.empty:
        print("Evaluation using Azure AI SDK failed. Exiting.")
        sys.exit(1)

    print("Running custom evaluations (Keyword Match)...")
    custom_eval_results = pf_results_df.apply(
        lambda row: keyword_match_evaluator(row['inputs.context'], row['inputs.answer']), 
        axis=1, 
        result_type='expand'
    )

    raw_final_df = pd.concat([
        pf_results_df.reset_index(drop=True),
        custom_eval_results.reset_index(drop=True)
    ], axis=1)
    
    print("\n--- Cleaning and structuring final results ---")
    final_df = flatten_and_clean_results(raw_final_df)

    metrics_cols = [col for col in final_df.columns if col != 'generated_text']
    metrics_df = final_df[metrics_cols]
    
    posts_cols = ['prompt_name', 'topic', 'generated_text']
    posts_df = final_df[posts_cols]

    output_dir = os.path.dirname(config.OUTPUT_FILE)
    metrics_output_file = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.csv")
    posts_output_file = os.path.join(output_dir, f"generated_posts_{timestamp}.csv")

    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(metrics_output_file, index=False)
    posts_df.to_csv(posts_output_file, index=False)
    
    print(f"\nEvaluation complete.")
    print(f"✅ Metrics-only results saved to: {metrics_output_file}")
    print(f"✅ Full generated posts saved to: {posts_output_file}")

    print("\n--- Evaluation Metrics Summary (Aggregated) ---")
    print(eval_summary_df.to_string())
    print("\n--- Sample of Metrics Table ---")
    print(metrics_df.head().to_string())
    
    # --- Run the validation and exit with an error if it fails ---
    failures = validate_metrics(metrics_df, config.EVALUATION_THRESHOLDS)
    if failures:
        print("\n\n--- 🚨 ACTION REQUIRED: One or more quality gates failed! ---")
        # Exit with a non-zero status code to fail the GitHub Action
        sys.exit(1) 
    else:
        print("\n\n--- ✅ All quality gates passed successfully! ---")


if __name__ == "__main__":
    main()
