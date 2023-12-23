from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas.langchain import RagasEvaluatorChain
import langchain

# Load & process
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from utils import *
# import re


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
import webbrowser

# Token counter
import tiktoken
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
from langchain.callbacks.manager import get_openai_callback
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
import os


# Newgen
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import pandas as pd
from tqdm import tqdm
from langchain.callbacks import get_openai_callback
#############################################################################













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










# ======================================
# CORE Functions
# ======================================

chroma_client = chromadb.PersistentClient()

def fetch_reference_from_pdf(uploaded_file, citation_number):
    # Save the uploaded PDF file temporarily
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, "temp.pdf")
    with open(temp_filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Open the saved PDF file
    pdf_document = fitz.open(temp_filepath)

    # Initialize the reference page content
    reference_page_content = ""

    # Iterate through all pages to find the REFERENCES section
    for current_page in pdf_document.pages():
        text = current_page.get_text()

        # Assuming the REFERENCES section starts with "REFERENCES" (modify as needed)
        if "REFERENCES" in text.upper():
            # Extract the content from the REFERENCES section
            reference_page_content = text
            break

    # Close the PDF document and remove the temporary files
    pdf_document.close()
    temp_dir.cleanup()

    # Extract the specific citation based on the citation_number
    citation_start = reference_page_content.find(f"[{citation_number}]")
    citation_end = reference_page_content.find("\n", citation_start)  # Assuming each citation is on a new line

    if citation_start != -1 and citation_end != -1:
        citation_text = reference_page_content[citation_start:citation_end].strip()
        return citation_text
    else:
        return None  # Return None if citation is not found
    
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)




# ======================================
# RETRIEVERS
# ======================================

# Preparation
corpus = pd.read_json(path_or_buf="data/corpus.jsonl", lines=True)

abstracts = list()
for a in corpus["abstract"]:
    text = ""
    for s in a:
        text += s + " "
    abstracts.append(text)

# Testing Data
train = pd.read_json("data/claims_train.jsonl", lines=True)

ids = list()
banyak_kontra = 0
for i in range(len(train)):
    doc_ids = train["evidence"].iloc[i].keys()

    labels = list()
    for evid in doc_ids:
        for j in range(len(train["evidence"].iloc[i][evid])):
            l = train["evidence"].iloc[i][evid][j]["label"]
            labels.append(l)
        
        if "CONTRADICT" not in labels:
            ids.append(i)
        else:
            banyak_kontra += 1

benchmark_data = train.iloc[ids]

embedder = OpenAIEmbeddings()

metadatas = list()
for i in range(len(corpus)):
    metadatas.append({
        "doc_id" : str(corpus["doc_id"].iloc[i]),
        "title": corpus["title"].iloc[i]
    })

ids = [f"abs{str(i)}" for i in range(len(corpus))]
len(metadatas), metadatas[:2], ids[:2]

docsearch = Chroma(
    client=chroma_client,
    collection_name="scifact_benchmark",
    embedding_function=embedder,
)




@st.cache_resource(ttl="1h")
def embed(uploaded_files):
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
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_API_KEY"])
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    return vectordb


def mmr(embed):
    mmr_retriever = embed.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return mmr_retriever




########################################
#          VANILLA RETRIEVER           #
########################################

benchmark_data = benchmark_data.reset_index(drop=True)
benchmark_data = benchmark_data.iloc[:100]









##################################################################################################


vanilla_tab, mmr_tab, multiquery_tab, compression_tab = st.tabs(["Vanilla", "MMR", "Multi-query", "Contextual Compression"])


with vanilla_tab:
    with st.sidebar:
        uploaded_files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            retriever = embed(uploaded_files)
            retriever = mmr(retriever)              #============ SUBJECT TO CHANGE ============#

        if "messages" not in st.session_state:
            st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

        if prompt := st.chat_input():
            st.session_state.messages.append(ChatMessage(role="user", content=prompt))
            st.chat_message("user").write(prompt)
            query = prompt

            if not uploaded_files:
                st.info("Please upload PDF documents to continue.")
                st.stop()

            with st.chat_message("assistant"):
                contexts = retriever.get_relevant_documents(prompt, k=3)
                stream_handler = StreamHandler(st.empty())
                chat_model = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], streaming=True, callbacks=[stream_handler])
                json_parser = SimpleJsonOutputParser()
                chain = prompt_formatted | chat_model | json_parser
                response = chain.invoke({"context": contexts_formatter(contexts), "question": st.session_state.messages, "chat_history": memory.buffer_as_messages})
                
                # Use 'context' as a reference for scoring
                reference = contexts_formatter(contexts)
                candidate = response["answer"]
                
                # Compute BLEU score
                bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing)

                # Compute ROUGE scores
                rouge_scores = scorer.score(reference, candidate)

                # Compute MMR
                # mrr = calculate_mean_reciprocal_rank([reference], [candidate])

                # === Display BLEU and ROUGE scores in the sidebar ====
                st.sidebar.write("\n")
                # st.sidebar.write(f"BLEU Score: {bleu_score:.4f}")
                # st.sidebar.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")
                # st.sidebar.write(f"MRR Score: {mrr:.4f}")

                eval_dict = {'source_documents': contexts, 'query': query, 'result': candidate}
                st.sidebar.write("**Information Accuracy**")
                # true, pred
                # st.sidebar.write(accuracy(contexts,contexts))
                
                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Retrieval Evaluation**")
                # RETRIEVAL EVAL
                eval_retrieve = {
                    m.name: RagasEvaluatorChain(metric=m) 
                    for m in [context_relevancy]
                }

                for name, eval_retrieve in eval_retrieve.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_retrieve(eval_dict)[score_name]}")       

                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Text-Generation Evaluation**")
                # GENERATION EVAL
                eval_generate = {
                    n.name: RagasEvaluatorChain(metric=n) 
                    for n in [faithfulness, answer_relevancy]

                }
                for name, eval_generate in eval_generate.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_generate(eval_dict)[score_name]}")

                    

                st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
                with st.expander("Reference"):
                    in_text_citation = response["in-text citation"]
                    
                    if not in_text_citation:
                        st.write("There are no citations.")
                    else:
                        try:
                            
                            citation_number = int(in_text_citation[1:-1])  # Extract the number inside brackets
                            uploaded_file = uploaded_files[0]  # Assuming the user uploaded only one file; adjust as needed
                            referencesget = fetch_reference_from_pdf(uploaded_file, citation_number)
                            
                            if referencesget is not None:
                                # extract the judul
                                # startl = referencesget.find('"')
                                # endl = referencesget.rfind('"')
                                # paper_title = referencesget[startl + 1:endl]
                                paper_title = referencesget
                                # st.write('debug0')
                                # st.write(paper_title)

                                # === scholar =
                                # # Display the citation without the number and create a hyperlink to search Google Scholar
                                # google_scholar_url = f"https://scholar.google.com/scholar?q={paper_title}"
                                # st.markdown(
                                #     f"<a href='{google_scholar_url}' onclick='window.open(\"{google_scholar_url}\", \"_blank\");'>{paper_title}</a>",
                                #     unsafe_allow_html=True
                                # )


                                # === search engine ===
                                # Display the citation without the number and create a hyperlink to search Google
                                google_search_url = f"https://www.google.com/search?q={paper_title}"
                                st.markdown(
                                    f"<a href='{google_search_url}' onclick='window.open(\"{google_search_url}\", \"_blank\");'>{paper_title}</a>",
                                    unsafe_allow_html=True
                                )

                            else:
                                st.write("Citation not found in the PDF.")
                        except ValueError:
                            st.write("Invalid citation format.")





with mmr_tab:
    with st.sidebar:
        uploaded_files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            retriever = embed(uploaded_files)
            retriever = mmr(retriever)              #============ SUBJECT TO CHANGE ============#

        if "messages" not in st.session_state:
            st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

        if prompt := st.chat_input():
            st.session_state.messages.append(ChatMessage(role="user", content=prompt))
            st.chat_message("user").write(prompt)
            query = prompt

            if not uploaded_files:
                st.info("Please upload PDF documents to continue.")
                st.stop()

            with st.chat_message("assistant"):
                contexts = retriever.get_relevant_documents(prompt, k=3)
                stream_handler = StreamHandler(st.empty())
                chat_model = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], streaming=True, callbacks=[stream_handler])
                json_parser = SimpleJsonOutputParser()
                chain = prompt_formatted | chat_model | json_parser
                response = chain.invoke({"context": contexts_formatter(contexts), "question": st.session_state.messages, "chat_history": memory.buffer_as_messages})
                
                # Use 'context' as a reference for scoring
                reference = contexts_formatter(contexts)
                candidate = response["answer"]
                
                # Compute BLEU score
                bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing)

                # Compute ROUGE scores
                rouge_scores = scorer.score(reference, candidate)

                # Compute MMR
                # mrr = calculate_mean_reciprocal_rank([reference], [candidate])

                # === Display BLEU and ROUGE scores in the sidebar ====
                st.sidebar.write("\n")
                # st.sidebar.write(f"BLEU Score: {bleu_score:.4f}")
                # st.sidebar.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")
                # st.sidebar.write(f"MRR Score: {mrr:.4f}")

                eval_dict = {'source_documents': contexts, 'query': query, 'result': candidate}
                st.sidebar.write("**Information Accuracy**")
                # true, pred
                # st.sidebar.write(accuracy(contexts,contexts))
                
                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Retrieval Evaluation**")
                # RETRIEVAL EVAL
                eval_retrieve = {
                    m.name: RagasEvaluatorChain(metric=m) 
                    for m in [context_relevancy]
                }

                for name, eval_retrieve in eval_retrieve.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_retrieve(eval_dict)[score_name]}")       

                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Text-Generation Evaluation**")
                # GENERATION EVAL
                eval_generate = {
                    n.name: RagasEvaluatorChain(metric=n) 
                    for n in [faithfulness, answer_relevancy]

                }
                for name, eval_generate in eval_generate.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_generate(eval_dict)[score_name]}")

                    

                st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
                with st.expander("Reference"):
                    in_text_citation = response["in-text citation"]
                    
                    if not in_text_citation:
                        st.write("There are no citations.")
                    else:
                        try:
                            
                            citation_number = int(in_text_citation[1:-1])  # Extract the number inside brackets
                            uploaded_file = uploaded_files[0]  # Assuming the user uploaded only one file; adjust as needed
                            referencesget = fetch_reference_from_pdf(uploaded_file, citation_number)
                            
                            if referencesget is not None:
                                # extract the judul
                                # startl = referencesget.find('"')
                                # endl = referencesget.rfind('"')
                                # paper_title = referencesget[startl + 1:endl]
                                paper_title = referencesget
                                # st.write('debug0')
                                # st.write(paper_title)

                                # === scholar =
                                # # Display the citation without the number and create a hyperlink to search Google Scholar
                                # google_scholar_url = f"https://scholar.google.com/scholar?q={paper_title}"
                                # st.markdown(
                                #     f"<a href='{google_scholar_url}' onclick='window.open(\"{google_scholar_url}\", \"_blank\");'>{paper_title}</a>",
                                #     unsafe_allow_html=True
                                # )


                                # === search engine ===
                                # Display the citation without the number and create a hyperlink to search Google
                                google_search_url = f"https://www.google.com/search?q={paper_title}"
                                st.markdown(
                                    f"<a href='{google_search_url}' onclick='window.open(\"{google_search_url}\", \"_blank\");'>{paper_title}</a>",
                                    unsafe_allow_html=True
                                )

                            else:
                                st.write("Citation not found in the PDF.")
                        except ValueError:
                            st.write("Invalid citation format.")



with multiquery_tab:
        with st.sidebar:
            uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
            )
            if uploaded_files:
                retriever = embed(uploaded_files)
                retriever = mmr(retriever)              #============ SUBJECT TO CHANGE ============#

            if "messages" not in st.session_state:
                st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

            for msg in st.session_state.messages:
                st.chat_message(msg.role).write(msg.content)

            if prompt := st.chat_input():
                st.session_state.messages.append(ChatMessage(role="user", content=prompt))
                st.chat_message("user").write(prompt)
                query = prompt

                if not uploaded_files:
                    st.info("Please upload PDF documents to continue.")
                    st.stop()

                with st.chat_message("assistant"):
                    contexts = retriever.get_relevant_documents(prompt, k=3)
                    stream_handler = StreamHandler(st.empty())
                    chat_model = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], streaming=True, callbacks=[stream_handler])
                    json_parser = SimpleJsonOutputParser()
                    chain = prompt_formatted | chat_model | json_parser
                    response = chain.invoke({"context": contexts_formatter(contexts), "question": st.session_state.messages, "chat_history": memory.buffer_as_messages})
                    
                    # Use 'context' as a reference for scoring
                    reference = contexts_formatter(contexts)
                    candidate = response["answer"]
                    
                    # Compute BLEU score
                    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing)

                    # Compute ROUGE scores
                    rouge_scores = scorer.score(reference, candidate)

                    # Compute MMR
                    # mrr = calculate_mean_reciprocal_rank([reference], [candidate])

                    # === Display BLEU and ROUGE scores in the sidebar ====
                    st.sidebar.write("\n")
                    # st.sidebar.write(f"BLEU Score: {bleu_score:.4f}")
                    # st.sidebar.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
                    # st.sidebar.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
                    # st.sidebar.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")
                    # st.sidebar.write(f"MRR Score: {mrr:.4f}")

                    eval_dict = {'source_documents': contexts, 'query': query, 'result': candidate}
                    st.sidebar.write("**Information Accuracy**")
                    # true, pred
                    # st.sidebar.write(accuracy(contexts,contexts))
                    
                    st.sidebar.write('\n ')
                    st.sidebar.write('\n ')

                    st.sidebar.write("**Retrieval Evaluation**")
                    # RETRIEVAL EVAL
                    eval_retrieve = {
                        m.name: RagasEvaluatorChain(metric=m) 
                        for m in [context_relevancy]
                    }

                    for name, eval_retrieve in eval_retrieve.items():   
                        score_name = f"{name}_score"
                        st.sidebar.write(f"{score_name}: {eval_retrieve(eval_dict)[score_name]}")       

                    st.sidebar.write('\n ')
                    st.sidebar.write('\n ')

                    st.sidebar.write("**Text-Generation Evaluation**")
                    # GENERATION EVAL
                    eval_generate = {
                        n.name: RagasEvaluatorChain(metric=n) 
                        for n in [faithfulness, answer_relevancy]

                    }
                    for name, eval_generate in eval_generate.items():   
                        score_name = f"{name}_score"
                        st.sidebar.write(f"{score_name}: {eval_generate(eval_dict)[score_name]}")

                        

                    st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
                    with st.expander("Reference"):
                        in_text_citation = response["in-text citation"]
                        
                        if not in_text_citation:
                            st.write("There are no citations.")
                        else:
                            try:
                                
                                citation_number = int(in_text_citation[1:-1])  # Extract the number inside brackets
                                uploaded_file = uploaded_files[0]  # Assuming the user uploaded only one file; adjust as needed
                                referencesget = fetch_reference_from_pdf(uploaded_file, citation_number)
                                
                                if referencesget is not None:
                                    # extract the judul
                                    # startl = referencesget.find('"')
                                    # endl = referencesget.rfind('"')
                                    # paper_title = referencesget[startl + 1:endl]
                                    paper_title = referencesget
                                    # st.write('debug0')
                                    # st.write(paper_title)

                                    # === scholar =
                                    # # Display the citation without the number and create a hyperlink to search Google Scholar
                                    # google_scholar_url = f"https://scholar.google.com/scholar?q={paper_title}"
                                    # st.markdown(
                                    #     f"<a href='{google_scholar_url}' onclick='window.open(\"{google_scholar_url}\", \"_blank\");'>{paper_title}</a>",
                                    #     unsafe_allow_html=True
                                    # )


                                    # === search engine ===
                                    # Display the citation without the number and create a hyperlink to search Google
                                    google_search_url = f"https://www.google.com/search?q={paper_title}"
                                    st.markdown(
                                        f"<a href='{google_search_url}' onclick='window.open(\"{google_search_url}\", \"_blank\");'>{paper_title}</a>",
                                        unsafe_allow_html=True
                                    )

                                else:
                                    st.write("Citation not found in the PDF.")
                            except ValueError:
                                st.write("Invalid citation format.")





with compression_tab:
    with st.sidebar:
        uploaded_files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            retriever = embed(uploaded_files)
            retriever = mmr(retriever)              #============ SUBJECT TO CHANGE ============#

        if "messages" not in st.session_state:
            st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

        if prompt := st.chat_input():
            st.session_state.messages.append(ChatMessage(role="user", content=prompt))
            st.chat_message("user").write(prompt)
            query = prompt

            if not uploaded_files:
                st.info("Please upload PDF documents to continue.")
                st.stop()

            with st.chat_message("assistant"):
                contexts = retriever.get_relevant_documents(prompt, k=3)
                stream_handler = StreamHandler(st.empty())
                chat_model = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], streaming=True, callbacks=[stream_handler])
                json_parser = SimpleJsonOutputParser()
                chain = prompt_formatted | chat_model | json_parser
                response = chain.invoke({"context": contexts_formatter(contexts), "question": st.session_state.messages, "chat_history": memory.buffer_as_messages})
                
                # Use 'context' as a reference for scoring
                reference = contexts_formatter(contexts)
                candidate = response["answer"]
                
                # Compute BLEU score
                bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing)

                # Compute ROUGE scores
                rouge_scores = scorer.score(reference, candidate)

                # Compute MMR
                # mrr = calculate_mean_reciprocal_rank([reference], [candidate])

                # === Display BLEU and ROUGE scores in the sidebar ====
                st.sidebar.write("\n")
                # st.sidebar.write(f"BLEU Score: {bleu_score:.4f}")
                # st.sidebar.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
                # st.sidebar.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")
                # st.sidebar.write(f"MRR Score: {mrr:.4f}")

                eval_dict = {'source_documents': contexts, 'query': query, 'result': candidate}
                st.sidebar.write("**Information Accuracy**")
                # true, pred
                # st.sidebar.write(accuracy(contexts,contexts))
                
                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Retrieval Evaluation**")
                # RETRIEVAL EVAL
                eval_retrieve = {
                    m.name: RagasEvaluatorChain(metric=m) 
                    for m in [context_relevancy]
                }

                for name, eval_retrieve in eval_retrieve.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_retrieve(eval_dict)[score_name]}")       

                st.sidebar.write('\n ')
                st.sidebar.write('\n ')

                st.sidebar.write("**Text-Generation Evaluation**")
                # GENERATION EVAL
                eval_generate = {
                    n.name: RagasEvaluatorChain(metric=n) 
                    for n in [faithfulness, answer_relevancy]

                }
                for name, eval_generate in eval_generate.items():   
                    score_name = f"{name}_score"
                    st.sidebar.write(f"{score_name}: {eval_generate(eval_dict)[score_name]}")

                    

                st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
                with st.expander("Reference"):
                    in_text_citation = response["in-text citation"]
                    
                    if not in_text_citation:
                        st.write("There are no citations.")
                    else:
                        try:
                            
                            citation_number = int(in_text_citation[1:-1])  # Extract the number inside brackets
                            uploaded_file = uploaded_files[0]  # Assuming the user uploaded only one file; adjust as needed
                            referencesget = fetch_reference_from_pdf(uploaded_file, citation_number)
                            
                            if referencesget is not None:
                                # extract the judul
                                # startl = referencesget.find('"')
                                # endl = referencesget.rfind('"')
                                # paper_title = referencesget[startl + 1:endl]
                                paper_title = referencesget
                                # st.write('debug0')
                                # st.write(paper_title)

                                # === scholar =
                                # # Display the citation without the number and create a hyperlink to search Google Scholar
                                # google_scholar_url = f"https://scholar.google.com/scholar?q={paper_title}"
                                # st.markdown(
                                #     f"<a href='{google_scholar_url}' onclick='window.open(\"{google_scholar_url}\", \"_blank\");'>{paper_title}</a>",
                                #     unsafe_allow_html=True
                                # )


                                # === search engine ===
                                # Display the citation without the number and create a hyperlink to search Google
                                google_search_url = f"https://www.google.com/search?q={paper_title}"
                                st.markdown(
                                    f"<a href='{google_search_url}' onclick='window.open(\"{google_search_url}\", \"_blank\");'>{paper_title}</a>",
                                    unsafe_allow_html=True
                                )

                            else:
                                st.write("Citation not found in the PDF.")
                        except ValueError:
                            st.write("Invalid citation format.")
































