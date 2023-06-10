import os
import pathlib

import openai
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
active_loop_org = os.getenv("ACTIVELOOP_ORG")


embeddings = OpenAIEmbeddings()

dataset_path = f"hub://{active_loop_org}/data"

docs_root = pathlib.Path("/home/anton/braintune/docs/docs")  # Change this to your docs root
docs = []
for doc in docs_root.rglob("*.md"):
    loader = UnstructuredMarkdownLoader(str(doc)).load()
    docs += loader

text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
texts = text_splitter.split_documents(docs)
db = DeepLake.from_documents(texts, embeddings, dataset_path=dataset_path, overwrite=True)
