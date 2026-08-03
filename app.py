import os
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# API KEY
# -------------------------------

groq_api_key = st.secrets["GROQ_API_KEY"]

# -------------------------------
# PERSONAS
# -------------------------------

SALES_PROMPTS = {

"PragyanAI Student Counselor":
"""
You are Aarav, an Academic Career Advisor.

Use ONLY the context below.

Context:
{context}

Guide students regarding fees, curriculum,
placements and projects.
""",

"PragyanAI Institutional / CoE Advisor":
"""
You are Dr. Kavita.

Use ONLY the context below.

Context:
{context}

Answer college partnership questions.
""",

"PragyanAI Enterprise AI & Placement Lead":
"""
You are Rohan.

Use ONLY the context below.

Context:
{context}

Answer hiring and enterprise AI questions.
"""
}

# -------------------------------
# EMBEDDINGS
# -------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# LOAD DOCUMENTS
# -------------------------------

@st.cache_resource
def load_vectorstore():

    docs=[]

    if os.path.exists("pragyan_faq_prices.xlsx"):

        df=pd.read_excel("pragyan_faq_prices.xlsx")

        for _,row in df.iterrows():

            content=" | ".join(
                [f"{c}:{v}" for c,v in row.items()]
            )

            docs.append(
                Document(page_content=content)
            )

    return FAISS.from_documents(docs,embeddings)

vectorstore=load_vectorstore()

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("⚙ Settings")

persona=st.sidebar.selectbox(
    "Choose Persona",
    list(SALES_PROMPTS.keys())
)

uploaded_files=st.sidebar.file_uploader(
    "Upload PDF/Excel",
    type=["pdf","xlsx","xls"],
    accept_multiple_files=True
)

# -------------------------------
# Process Uploaded Files
# -------------------------------

if uploaded_files:

    docs=[]

    for file in uploaded_files:

        if file.name.endswith(".pdf"):

            with open(file.name,"wb") as f:
                f.write(file.getbuffer())

            loader=PyPDFLoader(file.name)

            docs.extend(loader.load())

        else:

            df=pd.read_excel(file)

            for _,row in df.iterrows():

                docs.append(
                    Document(
                        page_content=" | ".join(
                            [f"{c}:{v}" for c,v in row.items()]
                        )
                    )
                )

    vectorstore=FAISS.from_documents(
        docs,
        embeddings
    )

    st.sidebar.success("Knowledge Base Updated")

# -------------------------------
# LLM
# -------------------------------

llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# -------------------------------
# CHAT HISTORY
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

# -------------------------------
# TITLE
# -------------------------------

st.title("🤖 PragyanAI Intelligent Assistant")

st.caption(
"Powered by Groq + LangChain + FAISS + Streamlit"
)

# -------------------------------
# DISPLAY CHAT
# -------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT
# -------------------------------

question=st.chat_input(
"Ask anything..."
)

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    retriever=vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    docs=retriever.invoke(question)

    context="\n".join(
        [doc.page_content for doc in docs]
    )

    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SALES_PROMPTS[persona].format(
                    context=context
                )
            ),
            (
                "human",
                "{input}"
            )
        ]
    )

    chain=prompt|llm

    response=chain.invoke(
        {
            "input":question
        }
    )

    answer=response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

# -------------------------------
# CLEAR CHAT
# -------------------------------

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages=[]

    st.rerun()
