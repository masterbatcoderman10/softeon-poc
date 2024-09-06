# this file will be used to load the data from the processed_data directory into the chroma database
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
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

### Setup the self-query retriever
self_query_store = Chroma(
    client=client,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="softeon_metadata_v3"
)

metadata_field_info = [
    AttributeInfo(
        name="file_name",
        description="The name of the file. You must only use this attribute if the filename is mentioned fully by the user in the query.",
        type="string",
    ),
    AttributeInfo(
        name="tag",
        description="The tag that is most appropriate for the text content. Please use the tags exactly as shown. One of the following: [WMS, Administrative, User Interface, Operational Processes, Automation, Inbound, Outbound, Integration, Compliance]",
        type="string",
    ),
    AttributeInfo(
        name="file_type",
        description="The type of the file. One of the following: [text, image]. `image` refers to the screens of the application. Use `image` filtering only when the query is asking about a screen of the application or related, in other cases use `text`.",
        type="string",
    ),
    AttributeInfo(
        name="page_number",
        description="The page number of the document. Please use this attribute only if the page number is mentioned in the query.",
        type="int",
    ),
    AttributeInfo(
        name="image_number",
        description="The image number of the document. Please use this attribute only if the image number is mentioned in the query and only for image `file_type`.",
        type="int",
    ),
]

document_content_description = "The documents contain information about a warehouse management system called Softeon."

llm = ChatOpenAI(model="gpt-4o-2024-08-06")

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=self_query_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)
