import os
import subprocess
import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from PIL import Image

# ============ CONFIG ============

# Load your GROQ API Key from environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")  # Use .env or set it in the cloud
MODEL_NAME = "llama3-8b-8192"

PDF_PATHS = {
    "Pharmacology": "pdfs/pharmacology.pdf",
    "Pain management": "pdfs/pain_management.pdf",
    "First aid": "pdfs/first_aid.pdf"
}

# ============ Functions ============

def load_medical_docs(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def build_qa_system(faiss_index):
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name=MODEL_NAME)
    return RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever(search_type="similarity", k=4))

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            st.warning("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
    return ""

def translate_to_arabic(text):
    try:
        return GoogleTranslator(source='auto', target='ar').translate(text)
    except Exception:
        return "تعذر الترجمة حالياً."

def generate_image_from_text(text):
    img = Image.new('RGB', (300, 100), color=(73, 109, 137))
    return img

# ============ Streamlit App ============

def main():
    st.set_page_config(page_title="Medical Chatbot (LLaMA 3 + RAG)", layout="centered")
    st.title("Medical Question Answering Chatbot (RAG + LLaMA 3)")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_book" not in st.session_state:
        st.session_state.selected_book = list(PDF_PATHS.keys())[0]

    st.subheader("Select Book:")
    for book_name in PDF_PATHS:
        if st.button(book_name):
            st.session_state.selected_book = book_name
            st.session_state.qa_chain = None
            st.session_state.chat_history = []

    if st.session_state.qa_chain is None:
        with st.spinner(f"Loading {st.session_state.selected_book}..."):
            docs = load_medical_docs(PDF_PATHS[st.session_state.selected_book])
            faiss_index = embed_documents(docs)
            st.session_state.qa_chain = build_qa_system(faiss_index)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask a medical question:")
    with col2:
        if st.button("Use Voice"):
            voice_input = recognize_speech()
            if voice_input:
                st.success(f"You said: {voice_input}")
                query = voice_input

    if query:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain.run(query)
            translation = translate_to_arabic(result)
            st.session_state.chat_history.append({
                "question": query,
                "answer": result,
                "translated": translation
            })

    st.markdown("---")
    st.subheader("Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.markdown(f"**ترجمة:** {chat['translated']}")

        if "image" in chat['answer'].lower() or "diagram" in chat['answer'].lower():
            if st.button(f"Generate Image for: {chat['question']}"):
                img = generate_image_from_text(chat['answer'])
                st.image(img, caption="Generated Illustration")

if __name__ == "__main__":
    main()


!streamlit run medical_rag_chatbot.py &/content/Logs.txt &
