# PromptOps: Investment Blog Post Generation

This project provides a structured framework for generating, evaluating, and comparing different versions of prompts for creating investment-themed blog posts. It uses the Azure AI SDK and `promptflow-evals` to score generated content on various metrics.

## Features

- **Prompt Versioning**: Easily manage and test multiple prompt variations from YAML files.
- **Centralized Configuration**: Tune LLM parameters (`temperature`, `max_tokens`, etc.) for all prompts in a single file.
- **Comprehensive Evaluation**: Automatically scores generated content on multiple metrics:
  - **Quality**: Coherence, Fluency, Relevance.
  - **Performance**: Latency, Token Usage, and Estimated Cost.
  - **Custom Metrics**: A custom keyword-matching evaluator.
- **Structured Output**: Generates a detailed `evaluation_results.csv` with all prompts, generated text, parameters, and scores for easy analysis.

## Project Structure

```
.
│
├── prompts/
│   └── investment_blog/            # Directory for prompt YAML files
│
├── data/
│   └── evaluation_topics.csv       # Topics and keywords for evaluation
│
├── outputs/                        # (Git-ignored) Directory for results
│   └── evaluation_results.csv
│
├── src/
│   ├── config.py                   # Loads environment variables
│   ├── llm_client.py               # Client for Azure OpenAI API
│   ├── prompt_config.py            # Centralized LLM parameters
│   └── run_evaluation.py           # Main script to run the pipeline
│
├── .gitignore                      # Specifies files for Git to ignore
├── env.example                     # Template for environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies. From the project root directory, run:

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file by copying the template and fill in your Azure credentials.

```bash
# In your terminal, copy the example file
cp env.example .env
```

Now, open the `.env` file and add your specific credentials. This file is listed in `.gitignore` and will not be committed to your repository.

```
# .env
AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
# ... and other credentials ...
```

## How to Run the Pipeline

With the setup complete, you can run the evaluation pipeline with a single command:

```bash
python -m src.run_evaluation
```

The script will:
1.  Load all prompts from the `prompts/` directory.
2.  Load all topics from `data/evaluation_topics.csv`.
3.  For each prompt and topic, generate a blog post using the Azure OpenAI API.
4.  Run all quality, performance, and custom evaluations.
5.  Save the complete results to `outputs/evaluation_results.csv`.

You can then open the CSV file in any spreadsheet program to analyze and compare the performance of your prompt versions.

## Customization

This project is designed to be easily extendable.

### Adding a New Prompt

1.  Create a new YAML file in `prompts/investment_blog/` (e.g., `v3.0_new_idea.yaml`).
2.  Define the `name`, `description`, and `template` in the file.
3.  Open `src/prompt_config.py` and add a new entry in the `PROMPT_PARAMS` dictionary with the same name to define its model parameters. If you don't add an entry, it will use `DEFAULT_PARAMS`.

That's it! The new prompt will be automatically included in the next evaluation run.

### Adding New Evaluation Topics

Simply open `data/evaluation_topics.csv` and add a new row with a `topic` and its associated `ground_truth_keywords`. 