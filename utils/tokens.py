from typing import Literal

import tiktoken
from llama_index.core.callbacks import TokenCountingHandler

# cost in dollars per thousand tokens
MODEL_COST = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def setup_token_counter(model_name):
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model_name).encode
    )
    token_counter.reset_counts()
    return token_counter


def calculate_cost(
    n_tokens: int, ttype: Literal["input", "output"], model_name=str
) -> float:
    n_tokens = n_tokens / 1000
    unit_cost = MODEL_COST[model_name][ttype]
    total_cost = n_tokens * unit_cost
    return total_cost
