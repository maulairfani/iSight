import tiktoken
from forex_python.converter import CurrencyRates
import fitz
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# ========================================
# Cost Calculator
# ========================================
def format_rupiah(amount):
    reversed_amount = str(amount)[::-1]
    formatted_amount = ",".join(reversed_amount[i:i+3] for i in range(0, len(reversed_amount), 3))
    result = "Rp" + formatted_amount[::-1]
    return result


def get_cost(
        prompt_formatted, #=prompt.format(context=contexts_formatter(contexts), question=question, chat_history=memory.buffer_as_messages),
        output_from_llm, #=dict_to_string(res_invoke)
        model="gpt-3.5-turbo"
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


    return f"""Tokens Used: {total_token}
        Prompt Tokens: {input_tokens_used}
        Completion Tokens: {output_tokens_used}
    Total Cost (USD): ${total_price_usd}
    Total Cost (IDR): Rp{total_price_idr}
    """


# ========================================
# Highlighting Sources
# ========================================
def get_matches(source, contexts, k=3):
    for i in range(k):
        idx_awal = contexts[i].page_content.find(source)
        if idx_awal != -1:
            idx_akhir = contexts[i].page_content[idx_awal:].find(".")
            idx_akhir += idx_awal
        
            match_ = contexts[i].page_content[idx_awal:idx_akhir]
            if len(match_) > 10:
                page_num = contexts[i].metadata["page"]
                return contexts[i].page_content[idx_awal:idx_akhir], page_num
    else:
        return None
    
def highlight_pdf(path, match_, page_num, output_path):
    pdf = fitz.open(path)
    page = pdf[page_num]

    matches = page.search_for(match_.replace("-\n", ""))

    for m in matches:
        page.add_highlight_annot(m)

    pdf.save(output_path)
    pdf.close()


# ========================================
# Get references
# ========================================
def get_reference(docs, in_text_citation):
    model = OpenAI()

    get_citation_template = """From the reference list below, rewrite the specified references!
    References:
    {references}

    Please rewrite the references related to the following numbers:
    {in_text_citation}
    """

    GET_CITATION_PROMPT = PromptTemplate.from_template(get_citation_template)

    get_reference_chain = chain = GET_CITATION_PROMPT | model

    for page_number in range(len(docs)):
        page = docs[page_number]
        text = page.page_content

        if "References" in text or "REFERENCES" in text:
            references_text = text.split("References")[1] if "References" in text else text.split("REFERENCES")[1]

    result = get_reference_chain.invoke({"references": references_text, "in_text_citation":in_text_citation})
    return result


# ========================================
# Get references
# ========================================
def contexts_formatter(contexts):
    result = ""
    for i in range(len(contexts)):
        result += f"{i+1}. {contexts[i].page_content}\n\n\n"
    return result

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