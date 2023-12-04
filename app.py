import langchain

# Load & process
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

# Vector store & embeddings
from langchain.vectorstores import Chroma
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

# For app
import streamlit as st

# Utils
from utils import *
import PyPDF2

# ========================================
# Hyperparameters
# ========================================

SALDO = 10000




# ========================================
# App
# ========================================

with st.sidebar:
    st.write("Sisa Saldo")
    st.success(f"**{format_rupiah(SALDO)}**")
    uploaded_file = st.file_uploader("Upload PDF(s)", type=("pdf")) 

st.title("Chatbot") 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # ========================================
    # Langchain process
    # ========================================
    if uploaded_file:
        # Load PDF
        st.write(uploaded_file)
        loader = PyPDFLoader(uploaded_file.name)
        st.write(dir(loader))
        docs = loader.load()

        # Split into chunks
        text_splitter = TokenTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 200
        )

        chunks = text_splitter.split_documents(docs)

        # Create embeddings from chunks
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(chunks, embeddings)

        # LLM
        chat_model = ChatOpenAI()

        # Prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(template),
            ]
        )

        # Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Chain
        json_parser = SimpleJsonOutputParser()
        chain = prompt | chat_model | json_parser
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = "response"
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(response)
        with st.expander("Source"):
            st.write("source")
    else:
        st.warning("Upload PDF dulu")
