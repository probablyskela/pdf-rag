import streamlit as st
from rag import RAG
from pypdf import PdfReader


st.title("PDF Retrieval-Augmented Generation App")

st.markdown(
    "This app allows you to ask questions results of which will be based on the pdf files you provide."
)

st.markdown(
    "To use this app, you will first need to create a Groq API key [here](https://console.groq.com/keys). "
    "After that, you can input your API."
    "Use the provided context to formulate the answer."
)

api_key = st.text_input("Groq API key:", placeholder="Your API key goes here.")

st.markdown("Now, upload your PDF file(-s) that will be used to answer your questions.")

files = st.file_uploader(
    "Upload your PDF files:",
    type="pdf",
    accept_multiple_files=True,
)

st.markdown("Finally, input your question and press `Send!`.")

question = st.text_input(
    "Ask a question:",
    placeholder="What are the names of the main characters in these books?",
    max_chars=100,
)

keywords = st.checkbox("Search using keywords.", value=True)
semantic = st.checkbox("Search using semantics.", value=True)


def on_click():
    if not api_key:
        st.error("Please input a Groq API key!")
        return

    if not files:
        st.error("Please upload PDF file(-s)!")
        return

    if not question:
        st.error("Please input your question!")
        return

    docs = []
    for file in files:
        reader = PdfReader(file)
        docs.append("\n".join(page.extract_text() for page in reader.pages))

    rag = RAG(api_key=api_key, docs=docs)

    if not rag.ping():
        st.error("An error has occurred! Please verify that your API key is correct and try again.")
        return

    answer = rag.answer_question(question=question, keywords=keywords, semantic=semantic)
    st.info(f"Answer:\n{answer}")


st.button("Send!", on_click=on_click)
