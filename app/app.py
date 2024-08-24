import chainlit as cl
from utils.rag_chain import chain_multimodal_rag


@cl.on_message
async def main(message: cl.Message):

    user_input = message.content
    response = await chain_multimodal_rag.ainvoke(
        {"input": user_input},
        config={"file_path": "data/chat_histories/sample_history.txt"}
    )
    await cl.Message(content=response).send()
