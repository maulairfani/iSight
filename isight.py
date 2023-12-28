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
# from langchain.vectorstores import Chroma
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
from PIL import Image
import webbrowser
import time


# Token counter
import tiktoken
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
from langchain.callbacks.manager import get_openai_callback
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
import os
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]



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

# Evaluation score
# Set up nltk BLEU and rouge scorer
smoothing = SmoothingFunction().method1  # Smoothing method for BLEU
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_mean_reciprocal_rank(ground_truths, ranked_list):
    for i, item in enumerate(ranked_list, 1):
        if item in ground_truths:
            return 1 / i
    return 0

refcomb = ["references", "referensi", "daftar pustaka", "sumber", "daftarpustaka","pustaka","bibliografi","bibliography","literatur","literature","workscited","works cited","reference"]

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

        
        if any(ref.lower() in text.lower() for ref in refcomb):
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

#===============COST================
from forex_python.converter import CurrencyRates
import json

def dict_to_string(input_dict):
    json_string = json.dumps(input_dict, indent=2)  # indent for pretty formatting (optional)
    return json_string








#################################################### 
# ==== DECLARE  =====#
####################################################


def has_highlighted_text(page):
    # Check if the page has any highlighted annotations
    return any(annot.type[0] == 8 for annot in page.annots())

def highlight_text_in_pdf(pdf_document, text_to_highlight):
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text_instances = page.search_for(text_to_highlight, hit_max=1)

        for text_instance in text_instances:
            page.add_highlight_annot(text_instance)






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






#===================================================
##################
#   RETRIEVERS   #
##################
#===================================================
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
llm = OpenAI(temperature=0, model="babbage-002")


 
# === RETRIEVAL ===
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
        # loader = fitz.open(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_API_KEY"])
    docsearch = DocArrayInMemorySearch.from_documents(splits, embeddings)

    return docsearch

def mmr(embed):
    mmr_retriever = embed.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 1})
    mmr_retriever = mmr_retriever.get_relevant_documents(question,k=1)
    return mmr_retriever

def vanilla(embed):
    vanilla_retriever = embed.similarity_search(question, k=1)
    return vanilla_retriever

def contextualcompression(embed):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=embed.as_retriever())
    context_compression = compression_retriever.get_relevant_documents(question, k=1)
    return context_compression

def multiquery(embed):
    mqmodeler = MultiQueryRetriever.from_llm(
        retriever=embed.as_retriever(), llm=llm
    )
    multiquery_retriever = mqmodeler.get_relevant_documents(query=question, k=1)
    return multiquery_retriever











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

if "saldo" not in st.session_state:
    st.session_state.saldo = 10000


with st.sidebar:

    st.write("**Saldoku**")
    saldo_placeholder = st.empty()  
    saldo_placeholder.write(st.session_state.saldo)

    uploaded_files = st.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        # for file in uploaded_files:
        #     page_count = get_page_count(uploaded_files)
        retrieval = embed(uploaded_files)
        # docs = []
        # temp_dir = tempfile.TemporaryDirectory()
        # for file in uploaded_files:
        #     temp_filepath = os.path.join(temp_dir.name, file.name)
        #     with open(temp_filepath, "wb") as f:
        #         f.write(file.getvalue())
        #     loader = PyPDFLoader(temp_filepath)
        #     docs.extend(loader.load())

        # # Split documents
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # splits = text_splitter.split_documents(docs)
        # # Create embeddings and store in vectordb
        # embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_API_KEY"])
        # retrieval = DocArrayInMemorySearch.from_documents(splits, embeddings)




# OPTIONS
with st.sidebar:
    select_retriever = 0    
    select_retriever = st.selectbox(
        "Select your retriever method",
        ("Vanilla", "Contextual Compression", "MMR","Multi-Query"),
        index = 0
    )
    display_eval = st.toggle("Display Evaluation Score")
    display_cost = st.toggle("Display Cost")
    display_references = st.toggle("Show References")


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
        stream_handler = StreamHandler(st.empty())
        chat_model = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], streaming=True, callbacks=[stream_handler])
        json_parser = SimpleJsonOutputParser()
        chain = prompt_formatted | chat_model | json_parser
        question = st.session_state.messages


        # RETRIEVER SELECTION #
        if select_retriever == 0:
            contexts = vanilla(retrieval)  
        elif select_retriever == 1:
            contexts = contextualcompression(retrieval)
        elif select_retriever == 2:
            contexts = mmr(retrieval)
        else:
            contexts = multiquery(retrieval)

        # contexts = retrieval.get_relevant_documents(prompt, k=3)
        # contexts = retrieval[0]
            
        def invoker(x,y,z):
            response = chain.invoke({"context": contexts_formatter(x), "question": y, "chat_history": z})
            return response

        response = invoker(contexts,question,memory.buffer_as_messages)
        
        # response = chain.invoke({"context": contexts_formatter(contexts), "question": question, "chat_history": memory.buffer_as_messages})
        # st.write(type(response),dir(response))
            
        # for s in chain.stream({"context": contexts_formatter(contexts), "question": question, "chat_history": memory.buffer_as_messages}):
        #     st.write(s, dir(s))
            # response = s.content


        
        # Use 'context' as a reference for scoring
        reference = contexts_formatter(contexts)
        candidate = response["answer"]
        # candidate = response.content


        # Subtractor
        curr = CurrencyRates()
        encoder = tiktoken.encoding_for_model("babbage-002")
        prmp = prompt_formatted.format(context=contexts_formatter(contexts), question=question, chat_history=memory.buffer_as_messages)
        output_from_llm=dict_to_string(candidate)
        input_tokens_used = len(encoder.encode(prmp)) + 7 # Jaga-jaga
        output_tokens_used = len(encoder.encode(output_from_llm))
        total_token = input_tokens_used + output_tokens_used
        input_price = round((0.0015/1000) * input_tokens_used, 8)
        output_price = round((0.002/1000) * output_tokens_used, 8)
        total_price_usd = round(input_price + output_price, 8)
        total_price_idr = curr.convert('USD', 'IDR', total_price_usd)
        st.session_state.saldo -= total_price_idr
        saldo_placeholder.text(st.session_state.saldo)




        # COSTING
        def get_cost(
                model="babbage-002", 
                prompt_formatted=prompt_formatted.format(context=contexts_formatter(contexts), question=question, chat_history=memory.buffer_as_messages),
                output_from_llm=dict_to_string(candidate)
        ):
            curr = CurrencyRates()
            encoder = tiktoken.encoding_for_model(model)
            input_tokens_used = len(encoder.encode(prompt_formatted)) + 7 # Jaga-jaga
            output_tokens_used = len(encoder.encode(output_from_llm))
            total_token = input_tokens_used + output_tokens_used

            input_price = round((0.0015/1000) * input_tokens_used, 8)
            output_price = round((0.002/1000) * output_tokens_used, 8)
            total_price_usd = round(input_price + output_price, 8)
            total_price_idr = curr.convert('USD', 'IDR', total_price_usd)
            st.session_state.saldo -= total_price_idr


            return f"""Tokens Used: {total_token}
                \nPrompt Tokens: {input_tokens_used}
                \nCompletion Tokens: {output_tokens_used}
            \nTotal Cost (USD): ${total_price_usd}
            \nTotal Cost (IDR): Rp{total_price_idr}
            """
        
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
        # st.sidebar.write("**Information Accuracy**")
        # true, pred
        # st.sidebar.write(accuracy(contexts,contexts))

        if display_eval:     
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

        if display_cost:
            # actcost = get_cost()
            # totcost =+ actcost
            st.sidebar.write('\n ')
            st.sidebar.write('\n ')
            st.sidebar.write('\n ')
            st.sidebar.write('**COST**')
            st.sidebar.write(get_cost())


        st.session_state.messages.append(ChatMessage(role="assistant", content=response["answer"]))
        if display_references:
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
                            st.write(
                                f"<a href='{google_search_url}' onclick='window.open(\"{google_search_url}\", \"_blank\");'>{paper_title}</a>",
                                unsafe_allow_html=True
                            )

                        else:
                            st.write("Citation not found in the PDF.")
                    except ValueError:
                        st.write("Invalid citation format.")
        





            #===== HIGHLIGHT =====#
            with st.expander("PDF Page"):
                st.write("debug0")
                pdf_document = fitz.open(uploaded_files[0])
                source_text = response["source"]
                if source_text:
                    highlight_text_in_pdf(pdf_document, source_text)
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        if has_highlighted_text(page):
                            pixmap = page.get_pixmap()
                            pil_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                            st.image(pil_image, caption=f"Page {page_num + 1}")
        
        # st.rerun()