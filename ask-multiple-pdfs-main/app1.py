import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template


# ---------- PDF PROCESSING ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.5
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Memory setup (persistent in session)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

    return conversation_chain


# ---------- HANDLE USER INPUT ----------
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return

    response = st.session_state.conversation({"question": user_question})
    answer = response["answer"]

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


# ---------- MAIN APP ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", layout="wide")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed successfully ✅")


if __name__ == "__main__":
    main()
