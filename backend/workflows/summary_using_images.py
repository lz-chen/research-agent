from pathlib import Path
import click
import logging

from llama_index.core import SimpleDirectoryReader

from prompts.prompts import SUMMARIZE_PAPER_PMT
from services.llms import new_mm_gpt4o, new_gpt4o
import sys
from llama_index.core import Settings
from utils.tokens import calculate_cost
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

Settings.llm = new_gpt4o()

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(Settings.llm.model).encode,
    # verbose=True
)
Settings.callback_manager = CallbackManager([token_counter])
Settings.llm.callback_manager = Settings.callback_manager

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_summary_from_gpt4o(img_dir):
    llm = new_mm_gpt4o()
    image_documents = SimpleDirectoryReader(img_dir).load_data()

    response = llm.complete(
        prompt=SUMMARIZE_PAPER_PMT,
        image_documents=image_documents,
    )
    return response.text.strip("```markdown").strip("```")


def save_summary_as_markdown(summary, output_file):
    with open(output_file, "w") as f:
        f.write(summary)
    logging.info(f"Summary saved to {output_file}")


def track_cost() -> None:
    # the token tracking is not accurate, esp when call is made in other modules
    logging.info(
        f"Embedding Tokens: {token_counter.total_embedding_token_count}\n"
        f"Prompt Tokens: {token_counter.prompt_llm_token_count}\n"
        f"Completion Tokens: {token_counter.completion_llm_token_count}\n"
        f"Total LLM Token Count: {token_counter.total_llm_token_count}"
    )

    total_pmt_cost = calculate_cost(
        token_counter.total_embedding_token_count
        + token_counter.prompt_llm_token_count,
        "input",
        Settings.llm.model,
    )
    total_comp_cost = calculate_cost(
        token_counter.completion_llm_token_count, "output", Settings.llm.model
    )
    logging.info(f"Total cost: USD {total_pmt_cost + total_comp_cost:.2f}")
    token_counter.reset_counts()


@click.command()
@click.option(
    "--folder_path",
    "-f",
    required=False,
    default="data/papers_image",
    help="Path to the folder containing images",
)
@click.option(
    "--output_dir",
    "-o",
    required=False,
    default="data/summaries_from_img_openai",
    help="Output markdown file for the summary",
)
def main(folder_path, output_dir):
    # get all subdirs
    for subdir in Path(folder_path).iterdir():
        summary_txt = get_summary_from_gpt4o(subdir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"{subdir.name}.md"
        save_summary_as_markdown(summary_txt, output_file)
        track_cost()


if __name__ == "__main__":
    main()
