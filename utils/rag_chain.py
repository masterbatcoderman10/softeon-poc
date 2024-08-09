from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import io
import re
import tiktoken
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import pickle
from langchain_core.documents import Document
import uuid
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain.tools import BaseTool, StructuredTool, tool
from .setup_retriever import retriever
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
import base64
load_dotenv()

model = ChatOpenAI(
    temperature=0, model="gpt-4o", max_tokens=1024)

# def plt_img_base64(img_base64):
#     """Disply base64 encoded string as image"""
#     # Create an HTML img tag with the base64 string as the source
#     image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
#     # Display the image by rendering the HTML
#     display(HTML(image_html))


def get_message_history(file_path: str) -> FileChatMessageHistory:
    # file_path = os.path.join('data', file_path)
    return FileChatMessageHistory(file_path)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        # Decode and get the first 8 bytes
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # print tokens
    # print(num_tokens_from_string(base64_string))
    return base64_string


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        doc = pickle.loads(doc)
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(400, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are an expert support assistant that works for Softeon Warehouse Management System\n"
            "You will be given a mixed of text, and image(s) usually of application screens.\n"
            "Use this information to provide information and support related to the user question. \n"
            "If no information is available, you will excuse yourself.\n"
            "Please do not return the images as markdown, only your response.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def contextualize_chain():

    sys_message = """
    - Based on the `chat_history` and the `input`, contextualize the query to a standalone form.
    - If there's no `chat_history`, the `input` should be returned as is.
    - Carefully consider how to best contextualize the query.
    - Keep your answers brief and concise.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_message),
            MessagesPlaceholder("chat_history"),
            ("user", "input")
        ]
    )

    contextual_chain = prompt | model | StrOutputParser()

    return contextual_chain


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    contextual_chain = contextualize_chain()

    # RAG pipeline
    rag_chain = (
        contextual_chain
        |
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        history_factory_config=[ConfigurableFieldSpec(
            id="file_path",
            annotation=str,
            name="File Path",
            description="Unique identifier of the history",
            default="",
            is_shared=True
        )]
    )

    chain = chain_with_history | StrOutputParser()
    return chain


if __name__ == "__main__":
    chain_multimodal_rag = multi_modal_rag_chain(retriever)
    response = chain_multimodal_rag.invoke(
        {"input": "Give me an overview of the warehouse management system"},
        config={"file_path": "data/chat_histories/sample_history.txt"}
    )
    print(response)
    # output = retriever.invoke(
    #     "Give me an overview of the warehouse management system")

    # for document in output:
    #     print(num_tokens_from_string(pickle.loads(document)))
    # Create RAG chain
    # chain_multimodal_rag = multi_modal_rag_chain(retriever)
