# this file will be used to load the data from the processed_data directory into the chroma database
import chromadb
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
import uuid
import os
from dotenv import load_dotenv
from .summarization import image_to_base64
from utils.setup_retriever import retriever
from utils.setup_retriever import self_query_retriever
import pickle
import json
load_dotenv()
id_key = "doc_id"
# splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=25)
llm = ChatOpenAI(model="gpt-3.5-turbo")


class TaggingSchema(BaseModel):
    explanation: str = Field(..., title="Explanation",
                             description="Detailed reasoning for the possible tags that fit the text content")
    tags: list[str] = Field(..., title="Tags",
                            description="Tags that are appropriate for the text content")


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


def load_retriever(retriever, text_dir, img_dir, img_summary_dir):
    text_data = get_text_docs(text_dir)
    img_data = get_images_n_summaries(img_dir, img_summary_dir)

    # add the image data to the retriever
    retriever.vectorstore.add_documents(img_data["summaries"])
    retriever.docstore.mset(list(zip(img_data["ids"], img_data["images"])))

    # add the text data to the retriever
    retriever.vectorstore.add_documents(text_data["sub_docs"])
    retriever.docstore.mset(list(zip(text_data["ids"], text_data["docs"])))


def produce_tags(text_content):

    tagging_prompt = """
    ###Objective###
    - Based on the text_content provide and appropriate tag for the text.

    ###Rules###
    - Select as many tags as you think are appropriate for the text.

    ###Tags###
    **WMS**
    - **Description**: Refers to documents related to the Warehouse Management System (WMS), including setup, configuration, and operational guidance specific to warehouse management processes.

    **Administrative**
    - **Description**: Covers documents focused on administrative tasks such as system configuration, user management, account management, and other setup-related activities within the WMS.

    **User Interface**
    - **Description**: Includes guides and documentation related to the customization, configuration, and design of the user interface within the WMS, focusing on UI objects, templates, and screen layouts.

    **Operational Processes**
    - **Description**: Pertains to documents that outline various operational tasks within the WMS, such as batch management, inbound and outbound logistics, and daily warehouse activities.

    **Automation**
    - **Description**: Refers to guides detailing automated processes within the WMS, including features that enhance efficiency by reducing manual intervention, such as auto-loading trucks and automated inbound inspections.

    **Inbound**
    - **Description**: Focuses on documentation related to inbound logistics processes, including the receiving, inspection, and handling of goods as they enter the warehouse.

    **Outbound**
    - **Description**: Covers documents related to outbound logistics processes, such as picking, packing, and shipping goods from the warehouse, as well as related operational tasks.

    **Integration**
    - **Description**: Documentation that deals with the integration of the WMS with other systems or processes, ensuring seamless operation and data flow across different platforms.

    **Compliance**
    - **Description**: Documentation that addresses regulatory or compliance-related requirements within the logistics and warehouse management processes, ensuring adherence to industry standards.### **WMS**

    `text_content`: {text_content}
    """

    tagging_prompt = ChatPromptTemplate.from_template(tagging_prompt)

    structured_llm = llm.with_structured_output(TaggingSchema)
    tagging_chain = tagging_prompt | structured_llm

    # print(f"Processing {text_content}")
    outputs = tagging_chain.invoke({"text_content": text_content})
    return outputs.dict()["tags"]


def tag_documents(data_dir, output_dir):

    texts = {}
    processed_texts = {}

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            texts[file] = f.read()

    for file, text in texts.items():
        tags = produce_tags(text)
        processed_texts[file] = {
            "file_name": file,
        }
        for i, tag in enumerate(tags):
            processed_texts[file][f"tag_{i}"] = tag
    
    os.makedirs(output_dir, exist_ok=True)
    # save json file
    with open(os.path.join(output_dir, "metadata_v1.json"), "w") as f:
        json.dump(processed_texts, f)

def create_documents_and_load(text_dir, metadata_file_path):

    with open(metadata_file_path, "r") as f:
        metadata = json.load(f)

    text_docs = {}

    for key, value in metadata.items():
        with open(os.path.join(text_dir, value["file_name"]), "r") as f:
            text = f.read()
            text_docs[key] = text

    documents = []
    # metadata = {}

    # for key, value in metadata.items():
    #     metadata[key] = {
    #         "file_name": value["file_name"],
    #     }
    #     for i, tag in enumerate(value["tags"]):
    #         metadata[key][f"tag_{i}"] = tag

    for key, text in text_docs.items():
        doc = Document(
            page_content=text,
            metadata=metadata[key]
        )
        documents.append(doc)

    self_query_retriever.vectorstore.add_documents(documents)



if __name__ == "__main__":
    # load_retriever("data/processed_data/text", "data/processed_data/images",
    #                "data/processed_data/image_summaries")
    # print("Data loaded successfully")
    tag_documents("data/temporary_data/text", "data/temporary_data/file_metadata")
    # create_documents_and_load("data/temporary_data/text", "data/temporary_data/file_metadata/metadata_v1.json")
