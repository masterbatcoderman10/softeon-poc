# this file will be used to load the data from the processed_data directory into the chroma database
import chromadb
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import uuid
import os
from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "data/db/chroma"
DOCSTORE_PATH = "data/db/docstore"

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DOCSTORE_PATH, exist_ok=True)

# chroma DB
client = chromadb.PersistentClient(CHROMA_PATH)
vector_store = Chroma(
    client=client,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="softeon"
)

# document store
docstore = LocalFileStore(DOCSTORE_PATH)
# basically what the metadata key for the ID of the document will be called
id_key = "doc_id"

# multi vector retriever
retriever = MultiVectorRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    id_key=id_key,
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
    }
)
