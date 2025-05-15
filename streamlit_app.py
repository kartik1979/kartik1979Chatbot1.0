from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
import os
import pandas as pd
import streamlit as st
import fitz
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CSV & PDF QA", layout="wide")
st.title("📄 CSV & PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])
question = st.text_input("Ask a question about the uploaded file")

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "csv":
        st.success("CSV file detected")
        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if "csv_agent" not in st.session_state:
            with open(tmp_path, "r", encoding="utf-8") as f:
                st.session_state.csv_agent = create_csv_agent(
                    ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY),
                    f,
                    verbose=False,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )

        if question:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.csv_agent.run(question)
                    st.write("### Answer:", response)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif file_type == "pdf":
        st.success("PDF file detected")
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if "pdf_chain" not in st.session_state:
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_text(text)

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_texts(splits, embeddings)

            llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
            st.session_state.pdf_chain = ConversationalRetrievalChain.from_llm(
                llm,
                vectorstore.as_retriever(),
                return_source_documents=True
            )
            st.session_state.chat_history = []

        if question:
            with st.spinner("Searching PDF..."):
                try:
                    result = st.session_state.pdf_chain({
                        "question": question,
                        "chat_history": st.session_state.chat_history
                    })
                    st.session_state.chat_history.append((question, result["answer"]))
                    st.write("### Answer:", result["answer"])
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Please upload a valid CSV or PDF file.")
