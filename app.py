from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


with st.sidebar:
    uploaded_files = st.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key="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", streaming=True, callbacks=[stream_handler])
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))