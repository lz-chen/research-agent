# bring in deps
import re
from pathlib import Path

import click
import qdrant_client
from llama_index.core.llms import LLM
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import (
    UnstructuredElementNodeParser,
)
from llama_index.readers.file import FlatReader, PDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import Settings
from config import settings
from prompts import LLAMAPARSE_INSTRUCTION
from services.llms import llm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


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


def parse_and_create_qe(file_name: str, llm: LLM):
    # use SimpleDirectoryReader to parse our file
    # file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[file_name],
                                      # file_extractor=file_extractor
                                      ).load_data()
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
    # Retrieve nodes (text) and objects (table)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    # # use llama parse to parse the file
    # parser = LlamaParse(
    #     api_key=settings.LLAMA_CLOUD_API_KEY,
    #     result_type="markdown",
    #     parsing_instruction=LLAMAPARSE_INSTRUCTION
    # )
    # documents = parser.load_data(file_name)  # each page will become a document
    # # Parse the documents using MarkdownElementNodeParser
    # node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
    # # Retrieve nodes (text) and objects (table)
    # nodes = node_parser.get_nodes_from_documents(documents)
    # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    # # use unsctructured element node parser to parse the file
    # reader = PDFReader()
    # documents = reader.load_data(Path(file_name))
    # node_parser = UnstructuredElementNodeParser()
    # raw_nodes = node_parser.get_nodes_from_documents(documents)
    # nodes, node_mappings = node_parser.get_base_nodes_and_mappings(
    #     raw_nodes
    # )

    # setup the vector store
    collection_name = re.sub(r'\W+', '_', file_name.split("/")[-1]).lower()
    vector_store = setup_vector_store(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        # nodes=base_nodes + objects,
        nodes=base_nodes,
        storage_context=storage_context
    )

    # index = VectorStoreIndex(
    #     nodes=nodes, storage_context=storage_context
    # )

    query_engine = index.as_query_engine(similarity_top_k=10,
                                         # node_postprocessors=[cohere_rerank]
                                         )

    return query_engine


@click.command()
@click.option("--file_name", "-f", required=True,
              help="Path to the file to parse and create the query engine")
def main(file_name: str):
    query_engine = parse_and_create_qe(file_name, llm_gpt4o)
    response = query_engine.query("What is the dataset used in this work?")
    print(response.metadata)
    print(response.response)


if __name__ == "__main__":
    main()
