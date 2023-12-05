from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st

import langchain

# Load & process
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from utils import *

# Vector store & embeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings

# Conversations
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.llms import OpenAI

# Post-processing
import fitz

# Token counter
import tiktoken
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
from langchain.callbacks.manager import get_openai_callback

# ======================================
# Prompt
# ======================================
# Template prompt
template = """Based ONLY on the context below, answer the following question according to the given format!
Context:
{context}

Question:
{question}

Format: a dictionary with keys
    "answer": (answer to the question)
    "in-text citation": (provide the in-text citation for the source within the context, usually in the form of [number], LEAVE IT BLANK IF NO IN-TEXT CITATION IS AVAILABLE IN THE ANSWER SOURCE)
    "source" : (one sentence that serves as the source of the answer, write it exactly as it appears in the context, including new line (-\n or \n) if any, LEAVE IT BLANK IF NO ANSWER IS AVAILABLE IN THE CONTEXT)
"""

# Prompt
prompt_formatted = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(template),
    ]
)

# memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ======================================
# App
# ======================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


with st.sidebar:
    uploaded_files = st.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        retriever = configure_retriever(uploaded_files)

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
        contexts = retriever.get_relevant_documents(prompt, k=3)
        stream_handler = StreamHandler(st.empty())
        chat_model = ChatOpenAI(openai_api_key="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", streaming=True, callbacks=[stream_handler])
        json_parser = SimpleJsonOutputParser()
        chain = prompt_formatted | chat_model | json_parser
        response = chain.invoke({"context": contexts_formatter(contexts), "question": st.session_state.messages, "chat_history": memory.buffer_as_messages})
        st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
        with st.expander("Reference"):
            st.write(response["in-text citation"])
        # st.expander("Highlighted Source")