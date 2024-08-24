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
llm = ChatOpenAI(model='gpt-4o')


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


def get_incomplete_files(dir_path, dst_path):

    # isolate files that are not present in dst_path but are there in dir_path, the filename is the same, but the extension is different
    img_files = [file for file in os.listdir(
        dir_path) if file.endswith(".png") or file.endswith(".jpg")]
    img_summaries = [file for file in os.listdir(
        dst_path) if file.endswith(".txt")]

    # compare using file extensions
    incomplete_files = [
        file for file in img_files if f"{file}.txt" not in img_summaries]
    return incomplete_files

# a function that takes in the directory of images, summarizes them, and saves the response as text files in a dst_path


def summarize_images(dir_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    incomplete_files = get_incomplete_files(dir_path, dst_path)
    print(f"Summarizing {len(incomplete_files)} images")
    for file in incomplete_files:
        file_path = os.path.join(dir_path, file)
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            try:
                response = get_image_summary(file_path)
                with open(os.path.join(dst_path, f"{file}.txt"), "w") as f:
                    f.write(response)
            except Exception as e:
                print(e)
                continue


# if __name__ == "__main__":
#     llm = ChatOpenAI(model='gpt-4o')
#     summarize_images("data/processed_data/images",
#                      "data/processed_data/image_summaries")
