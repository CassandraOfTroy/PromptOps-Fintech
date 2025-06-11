# Centralized configuration for prompt parameters

# Default parameters to be used if a specific prompt version is not defined below.
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 800,
    "top_p": 1.0,
}

# Parameters for specific prompt versions, keyed by the prompt name from the YAML file.
PROMPT_PARAMS = {
    "v1.0_simple_request": {
        "temperature": 0.7,
        "max_tokens": 800,
    },
    "v1.1_with_persona": {
        "temperature": 0.75,
        "max_tokens": 800,
    },
    "v1.2_structured_output": {
        "temperature": 0.7,
        "max_tokens": 900,
    },
    "v2.0_expert_tone": {
        "temperature": 0.6,
        "max_tokens": 1000,
        "top_p": 0.95,
    },
}

def get_params_for_prompt(prompt_name: str) -> dict:
    """
    Returns the parameters for a given prompt name, falling back to defaults.
    """
    params = DEFAULT_PARAMS.copy()
    specific_params = PROMPT_PARAMS.get(prompt_name, {})
    params.update(specific_params)
    return params
