import streamlit as st
import os
from PyPDF2 import PdfReader, PdfWriter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["HF_TOKEN", "LANGCHAIN_API_KEY", "GROQ_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"Missing environment variable: {var}")
        st.stop()

os.environ['LANGCHAIN_PROJECT'] = "RAG DOCUMENT Q&A Groq"
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Initialize the Gemini LLM
@st.cache_resource
def init_llm():
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="Gemma2-9b-It")

llm = init_llm()

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Initialize the embeddings model
@st.cache_resource
def init_embeddings():
    try:
        return HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError as e:
        st.error(f"ImportError: {str(e)}. Please ensure that all required dependencies are installed.")
        st.stop()

embeddings = init_embeddings()

# Function to decrypt PDF files (no password required)
def decrypt_pdf(input_path, output_path):
    with open(input_path, "rb") as input_file:
        reader = PdfReader(input_file)
        if reader.is_encrypted:
            reader.decrypt('')
            writer = PdfWriter()
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                writer.add_page(page)
            with open(output_path, "wb") as output_file:
                writer.write(output_file)
            return output_path
        else:
            return input_path

# Function to create the vector store from PDF documents
@st.cache_data
def create_vector_store():
    try:
        directory_path = "./ResearchPapers"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        docs = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.pdf'):
                input_path = os.path.join(directory_path, file_name)
                decrypted_path = decrypt_pdf(input_path, input_path + "_decrypted.pdf")
                loader = PyPDFLoader(decrypted_path)
                docs.extend(loader.load())

        if not docs:
            st.error("No documents found. Please ensure the 'ResearchPapers' directory contains PDF files.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        if not final_documents:
            st.error("Document splitting failed. No chunks were created from the documents.")
            return None

        return FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Streamlit UI setup
st.title("RAG Document Q&A with Groq and Gemini")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_vector_store()

if st.button("Create Document Embeddings"):
    with st.spinner("Creating vector database..."):
        st.session_state.vector_store = create_vector_store()
    if st.session_state.vector_store:
        st.success("Vector database is ready")

user_prompt = st.text_input("Enter your query from the research papers")

if user_prompt:
    if st.session_state.vector_store:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            process_time = time.process_time() - start

            if response.get('answer'):
                st.write(response['answer'])
            else:
                # Fallback to using the Gemini model for an answer if no answer is found in the documents
                fallback_response = llm.invoke({'input': user_prompt})
                st.write(fallback_response['output'])

            st.info(f"Response Time: {process_time:.2f} seconds")

            with st.expander("Document Similarity Search"):
                seen_contexts = set()
                for i, doc in enumerate(response.get('context', [])):
                    if doc.page_content not in seen_contexts:
                        st.write(doc.page_content)
                        seen_contexts.add(doc.page_content)
                        st.write("--------------")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please create document embeddings first.")

# PDF Uploader
st.sidebar.title("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if not os.path.exists("./ResearchPapers"):
        os.makedirs("./ResearchPapers")

    for uploaded_file in uploaded_files:
        with open(os.path.join("./ResearchPapers", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("Uploaded successfully.")
