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
from .summarization import image_to_base64
import pickle
load_dotenv()


CHROMA_PATH = "data/db/chroma"
DOCSTORE_PATH = "data/db/docstore"

os.makedirs(CHROMA_PATH, exist_ok=True)

# chroma DB
client = chromadb.PersistentClient(CHROMA_PATH)
vector_store = Chroma(
    client=client,
    embedding_function=OpenAIEmbeddings(),
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
    id_key=id_key
)

# splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=25)


def get_text_docs(dir_path):
    loaders = []
    docs = []
    sub_docs = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        loaders.append(TextLoader(file_path))

    for loader in loaders:
        docs.extend(loader.load())

    doc_ids = [str(uuid.uuid4()) for _ in docs]
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    docs = [doc.page_content for doc in docs]
    docs = [pickle.dumps(doc) for doc in docs]

    return {
        "docs": docs,
        "sub_docs": sub_docs,
        "ids": doc_ids
    }


def get_images_n_summaries(img_dir, img_summary_dir):

    base64_images = []
    summaries = []

    img_files = sorted(os.listdir(img_dir))
    img_summary_files = sorted(os.listdir(img_summary_dir))

    for file in img_files:
        file_path = os.path.join(img_dir, file)
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            image = image_to_base64(file_path)
            base64_images.append(image)

    img_ids = [str(uuid.uuid4()) for _ in base64_images]
    # the summaries are in .txt files their string content can be read and appended to the summaries list
    for file, img_id in zip(img_summary_files, img_ids):
        file_path = os.path.join(img_summary_dir, file)
        with open(file_path, "r") as f:
            summary = f.read()
            summary_docs = Document(
                page_content=summary,
                metadata={id_key: img_id}
            )
            summaries.append(summary_docs)

    base64_images = [pickle.dumps(img) for img in base64_images]

    return {
        "images": base64_images,
        "summaries": summaries,
        "ids": img_ids
    }


def load_retriever(text_dir, img_dir, img_summary_dir):
    text_data = get_text_docs(text_dir)
    img_data = get_images_n_summaries(img_dir, img_summary_dir)

    # add the image data to the retriever
    retriever.vectorstore.add_documents(img_data["summaries"])
    retriever.docstore.mset(list(zip(img_data["ids"], img_data["images"])))

    # add the text data to the retriever
    retriever.vectorstore.add_documents(text_data["sub_docs"])
    retriever.docstore.mset(list(zip(text_data["ids"], text_data["docs"])))


if __name__ == "__main__":
    load_retriever("data/processed_data/text", "data/processed_data/images",
                   "data/processed_data/image_summaries")
    print("Data loaded successfully")
