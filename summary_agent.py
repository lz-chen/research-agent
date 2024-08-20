# TODO:
# - paper to pdf should happen before this script
# - avoid re-ingesting nodes to vector store ✅
# - investigate the index nodes and objects
# - use unstructured for parsing might be faster

import re
from pathlib import Path
from typing import Optional

import click
import qdrant_client
from llama_index.core.agent import ReActAgent, ReActChatFormatter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import (
    UnstructuredElementNodeParser,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.readers.file import FlatReader, PDFReader
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import Settings
from config import settings
from prompts import LLAMAPARSE_INSTRUCTION, SUMMARIZE_PAPER_PMT_REACT
from services.llms import llm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


def fname_to_collection_name(fname: str):
    return re.sub(r'\W+', '_', fname.split("/")[-1]).lower()


def setup_vector_store(collection_name: str):
    """Set up the Qdrant vector store. vector store is used to
    store the embeddings of the source nodes."""
    client = qdrant_client.QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    aclient = qdrant_client.AsyncQdrantClient(
        host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
    )
    return QdrantVectorStore(
        client=client, aclient=aclient, collection_name=collection_name
    )


def setup_doc_store(namespace: str):
    """Set up the MongoDB document store. docstore is used to store the source node data."""
    return RedisDocumentStore.from_host_and_port(
        settings.REDIS_HOST,
        settings.REDIS_PORT,
        # namespace="SourceDocs",
        namespace=namespace
    )


def setup_index_store(namespace: str, collection_suffix: str):
    """Set up the MongoDB index store. index store is used to store the metadata of the index creation."""
    return RedisIndexStore.from_host_and_port(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        namespace=namespace,
        # collection_suffix="/index"
        collection_suffix=collection_suffix
    )


def paper2md(fname: Path, output_dir: Path, langs: Optional[list] = ["English"], max_pages: Optional[int] = None,
             batch_multiplier: Optional[int] = 1, start_page: Optional[int] = None):
    # https://github.com/VikParuchuri/marker/blob/master/convert_single.py#L8
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    from marker.output import save_markdown
    import time

    model_lst = load_all_models()
    start = time.time()
    full_text, images, out_meta = convert_single_pdf(fname.as_posix(), model_lst, max_pages=max_pages, langs=langs,
                                                     batch_multiplier=batch_multiplier, start_page=start_page)

    name = fname.name
    subfolder_path = save_markdown(output_dir.as_posix(), name, full_text, images, out_meta)

    logging.info(f"Saved markdown to the '{subfolder_path}' folder")
    logging.debug(f"Total time: {time.time() - start}")
    return subfolder_path


def parse_and_create_qe(file_name: Path, llm: LLM, force_reparse: bool = False, force_reingest: bool = False):
    # use SimpleDirectoryReader to parse our file
    md_output_dir = file_name.parents[1]

    if len(list(md_output_dir.joinpath(file_name.stem).glob("*.md"))) and not force_reparse:
        logging.info(f"Markdown file already exists for '{file_name}'")
        md_folder = md_output_dir.joinpath(file_name.stem)
    else:
        logging.info(f"Converting '{file_name}' to markdown")
        md_folder = paper2md(Path(file_name), md_output_dir)
    documents = SimpleDirectoryReader(input_dir=md_folder,
                                      # file_extractor=file_extractor
                                      required_exts=[".md"], filename_as_id=True,
                                      ).load_data()

    # setup the vector store
    collection_name = fname_to_collection_name(file_name.as_posix())
    vector_store = setup_vector_store(collection_name)
    doc_store = setup_doc_store(namespace=f"ra_qe_{collection_name}")
    index_store = setup_index_store(namespace=f"ra_qe_{collection_name}",
                                    collection_suffix=f"/index_ra_qe_{collection_name}")
    storage_context = StorageContext.from_defaults(vector_store=vector_store,
                                                   docstore=doc_store,
                                                   index_store=index_store)

    # if force_reingest:
    #     vector_store.clear()
    #     ref_docs = doc_store.get_all_ref_doc_info()
    #     for d in ref_docs:
    #         for id_ in d.node_ids:
    #             doc_store.delete_ref_doc(id_)

    # setup ingest pipeline
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
    pipeline = IngestionPipeline(
        transformations=[node_parser],
        vector_store=vector_store,
        # cache=cache,
        docstore=doc_store,
    )
    nodes = pipeline.run(documents=documents) # this nodes doesn't contain objects???

    # # Retrieve nodes (text) and objects (table)
    # nodes = node_parser.get_nodes_from_documents(documents)
    # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    index = VectorStoreIndex(
        # nodes=base_nodes + objects,
        nodes=nodes,
        storage_context=storage_context
    )
    index.set_index_id(collection_name)

    query_engine = index.as_query_engine(similarity_top_k=10)

    return query_engine


def save_paper_sumamry(paper_name: str, summary_text: str, output_dir: str = "./data/summaries"):
    """
    Save the paper summary to a markdown file
    :param paper_name: name of the paper, will be part of the file name
    :param summary_text: summary text to save
    :param output_dir: the output directory to save the summary
    :return:
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_file = Path(output_dir).joinpath(f"{paper_name}_summary.md")
    with open(summary_file, "w") as f:
        f.write(summary_text)
    logging.info(f"Saved summary to '{summary_file}'")


def create_agent(file_name: str):
    logging.info(f"Creating query engine for '{file_name}'")
    query_engine = parse_and_create_qe(Path(file_name), llm_gpt4o)
    logging.info(f"Creating agent for querying '{file_name}'")
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=f"index_{fname_to_collection_name(file_name)}",
            description=(
                f"This index contains information about paper {file_name}, "
                "Use a detailed plain text question as input to the tool to query information in the table."
            ),
        ),
    )
    save_tool = FunctionTool.from_defaults(fn=save_paper_sumamry)

    # chat_formatter = ReActChatFormatter(context=SUMMARIZE_PAPER_PMT)
    agent = ReActAgent.from_tools([query_tool, save_tool], llm=llm_gpt4o,
                                  # react_chat_formatter=chat_formatter,
                                  max_iterations=30,
                                  verbose=True)
    agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(SUMMARIZE_PAPER_PMT_REACT)})
    agent.chat(f"I want a summary of paper {file_name}")


@click.command()
@click.option("--file_dir", "-d", required=True,
              help="Path to the directory that contains file to parse and create the query engine")
def main(file_dir: str):
    for f in Path(file_dir).rglob("*.pdf"):
        if f.parents[1].joinpath("summaries").joinpath(f"{f.stem}_summary.md").exists():
            logging.info(f"Summary already exists for '{f}', skip generation")
            continue
        logging.info(f"Creating agent for '{f}' to generate summary")
        create_agent(f.as_posix())

    # create_agent(file_name)
    # query_engine = parse_and_create_qe(Path(file_name), llm_gpt4o)
    # response = query_engine.query("What is the dataset used in this work?")
    # print(response.metadata)
    # print(response.response)


if __name__ == "__main__":
    main()
