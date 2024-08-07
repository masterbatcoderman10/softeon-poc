from langchain_openai import ChatOpenAI
import base64
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain.tools import BaseTool, StructuredTool, tool
load_dotenv()
# a function that takes in an image path, converts it to base64 and returns the response from the model


def image_to_base64(image_path):
    """Converts an image to a Base64 encoded string.

    Args:
      image_path: The path to the image file.

    Returns:
      The Base64 encoded string of the image.
    """

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

        return encoded_string.decode('utf-8')


def get_image_summary(image_path):
    image = image_to_base64(image_path)
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": "Provide a detailed summary of the provided image."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        },
    ])

    response = llm.invoke([message])

    return response.content

# a function that takes in the directory of images, summarizes them, and saves the response as text files in a dst_path


def summarize_images(dir_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            response = get_image_summary(file_path)
            with open(os.path.join(dst_path, f"{file}.txt"), "w") as f:
                f.write(response)


if __name__ == "__main__":
    llm = ChatOpenAI(model='gpt-4o')
    summarize_images("data/processed_data/images",
                     "data/processed_data/image_summaries")
