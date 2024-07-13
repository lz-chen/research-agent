import tiktoken


def count_tokens(input_string: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = tokenizer.encode(input_string)

    return len(tokens)


def calculate_cost(input_string: str, cost_per_million_tokens: float = 5) -> float:
    num_tokens = count_tokens(input_string)

    total_cost = (num_tokens / 1_000_000) * cost_per_million_tokens

    return total_cost

# Example usage:
# input_string = "What's the difference between
# beer nuts and deer nuts? Beer nuts are about 5 dollars. Deer nuts are just under a buck."
# cost = calculate_cost(input_string)
# print(f"The total cost for using gpt-4o is: $US {cost:.6f}")
