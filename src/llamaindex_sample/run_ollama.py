from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.postprocessor.cohere_rerank import CohereRerank

import os
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

os.environ["TRANSFORMERS_OFFLINE"] = "0"
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


reader = SimpleDirectoryReader("/Users/lzchen/PycharmProjects/llamaindex-sample/data/admin")
docs = reader.load_data()

if not os.path.exists("storage"):
    index = VectorStoreIndex.from_documents(docs)
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist("./storage")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")

# generate question regarding topic
prompt_str1 = "What is the process for {topic}"
prompt_tmpl1 = PromptTemplate(prompt_str1)
# use HyDE to hallucinate answer.
prompt_str2 = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{query_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)
prompt_tmpl2 = PromptTemplate(prompt_str2)

llm = Settings.llm
retriever = index.as_retriever(similarity_top_k=5)
p = QueryPipeline(
    chain=[prompt_tmpl1, llm, prompt_tmpl2, llm, retriever], verbose=True
)
nodes = p.run(topic="college")
print(len(nodes))

# resp = llm.complete("Who is Paul Graham?")
# print(resp)
