import streamlit as st
import os
import re
import json
import glob
from typing import List, Dict, Any
import time

# LangChain & dependencies
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Use Google AI Embeddings
from langchain_community.vectorstores import FAISS      # FAISS is the vector store using the embeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI # Use Google LLM
from dotenv import load_dotenv

st.set_page_config(layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    /* Force dark theme on the entire app */
    body {
        background-color: #1a1a1a !important;
        font-family: 'Roboto', 'Arial', sans-serif !important;
        color: #d9d9d9 !important;
    }
    div[data-testid="stAppViewContainer"] {
        background-color: #1a1a1a !important;
    }
    h1 {
        color: #00c4b4 !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 1.5rem !important;
    }
    h2, h3 {
        color: #00c4b4 !important;
        font-weight: 600 !important;
    }
    div.stMarkdown {
        background-color: #2b2b2b !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 1rem !important;
        color: #d9d9d9 !important;
    }
    div.stTextInput > div > div > input {
        background-color: #2b2b2b !important;
        border: 2px solid #00c4b4 !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        font-size: 1rem !important;
        color: #d9d9d9 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #00a69a !important;
        box-shadow: 0 0 8px rgba(0, 196, 180, 0.3) !important;
    }
    div.stButton > button {
        background-color: #00c4b4 !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: background-color 0.3s ease, transform 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #00a69a !important;
        transform: translateY(-2px) !important;
    }
    div.stSpinner > div {
        color: #00c4b4 !important;
    }
    div.stSuccess {
        background-color: #1c3a38 !important;
        color: #00c4b4 !important;
        border-left: 4px solid #00c4b4 !important;
        padding: 0.8rem !important;
        border-radius: 4px !important;
    }
    div.stWarning {
        background-color: #3a2f1c !important;
        color: #ffab40 !important;
        border-left: 4px solid #ffab40 !important;
        padding: 0.8rem !important;
        border-radius: 4px !important;
    }
    div.stError {
        background-color: #3a1c1c !important;
        color: #ff5252 !important;
        border-left: 4px solid #ff5252 !important;
        padding: 0.8rem !important;
        border-radius: 4px !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #2b2b2b !important;
        color: #d9d9d9 !important;
        padding: 1.5rem !important;
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #00c4b4 !important;
    }
    div[data-testid="stSidebar"] div.stMarkdown,
    div[data-testid="stSidebar"] div.stSuccess,
    div[data-testid="stSidebar"] div.stWarning,
    div[data-testid="stSidebar"] div.stError {
        background-color: #3a3a3a !important;
        color: #d9d9d9 !important;
        border-radius: 4px !important;
        padding: 0.8rem !important;
    }
    div.stExpander {
        background-color: #2b2b2b !important;
        border: 1px solid #444444 !important;
        border-radius: 6px !important;
        margin-bottom: 1rem !important;
        color: #d9d9d9 !important;
    }
    div.stExpander > div > div {
        background-color: #3a3a3a !important;
        padding: 0.8rem !important;
        color: #d9d9d9 !important;
    }
    hr {
        border: 1px solid #00c4b4 !important;
        margin: 1.5rem 0 !important;
    }
    div.stProgress > div > div {
        background-color: #00c4b4 !important;
    }
    p, span, div {
        color: #d9d9d9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables (API keys)
load_dotenv()

# --- Configuration ---
DATA_DIR = "./mimic-iv-ext-direct-1.0.0" # Adjust if your path differs

# Google AI Embeddings Configuration
GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004" # Google embedding model

# Google LLM Configuration
GOOGLE_LLM_MODEL = "gemini-2.0-flash" # Or other compatible Gemini model

# RAG Configuration
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
SEARCH_K = 3 # Number of documents to retrieve

# --- API Key Loading and Validation ---
GOOGLE_API_KEY = "AIzaSyAiiQpVNO4-EH1lt9yhkd-ZE_zUlqGx2sw"

# --- Dynamic Vector Store Path based on Embedding Model ---
# Replace characters invalid for directory names
safe_embedding_model_name = GOOGLE_EMBEDDING_MODEL.replace("/", "_").replace(":", "_")
VECTOR_STORE_PATH = f"faiss_vectorstore_{safe_embedding_model_name}"



# --- Helper Functions (Data Loading, Preprocessing, Chunking - Keep as before) ---

def extract_dict_text(obj: Any) -> str:
    """ Recursively extracts text from nested dict/list structures. """
    out = []
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, (dict, list)):
                out.append(extract_dict_text(val))
            else:
                out.append(str(val))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                out.append(extract_dict_text(item))
            else:
                out.append(str(item))
    else:
        out.append(str(obj))
    return " ".join(filter(None, out)) # Filter out potential None values

def expand_synonyms(text: str) -> str:
    """ Basic expansions for domain abbreviations. """
    expansions = {
        r"\bHF\b": "heart failure", r"\bHFrEF\b": "heart failure with reduced ejection fraction",
        r"\bCHF\b": "congestive heart failure", r"\bHTN\b": "hypertension",
        r"\bDM2?\b": "type 2 diabetes mellitus", r"\bT2DM\b": "type 2 diabetes mellitus",
        r"\bCAD\b": "coronary artery disease", r"\bMI\b": "myocardial infarction",
        r"\bCABG\b": "coronary artery bypass graft", r"\bPCI\b": "percutaneous coronary intervention",
        r"\bAFib\b": "atrial fibrillation", r"\bCKD\b": "chronic kidney disease",
        r"\bESRD\b": "end-stage renal disease", r"\bPE\b": "pulmonary embolism",
        r"\bDVT\b": "deep vein thrombosis", r"\bCOPD\b": "chronic obstructive pulmonary disease",
        r"\bUA\b": "unstable angina", r"\bNSTEMI\b": "non-ST elevation myocardial infarction",
        r"\bSTEMI\b": "ST elevation myocardial infarction",
        r"LVEF\s*<\s*40%": "heart failure with reduced ejection fraction"
    }
    processed_text = text
    for pattern, repl in expansions.items():
        processed_text = re.sub(pattern, repl, processed_text, flags=re.IGNORECASE)
    return processed_text

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_mimic_finished(data_dir: str) -> List[Document]:
    """ Loads and preprocesses documents from MIMIC 'Finished' directory. """
    st.write(f"Checking directory: {data_dir}")
    finished_dir = os.path.join(data_dir, "Finished")
    if not os.path.isdir(finished_dir):
        st.error(f"Error: Data directory not found: {finished_dir}")
        st.info(f"Please ensure the MIMIC-IV-Ext data is unzipped at: {DATA_DIR}")
        return []

    json_files = glob.glob(os.path.join(finished_dir, "**", "*.json"), recursive=True)
    docs = []
    st.write(f"Found {len(json_files)} JSON files.")

    progress_bar = st.progress(0, text="Loading and preprocessing files...")
    total_files = len(json_files)
    loaded_count = 0

    for i, path in enumerate(json_files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue # Skip files that can't be opened or parsed

        text_parts = []
        metadata = {"source": os.path.basename(path)}

        for k, v in data.items():
            if isinstance(v, dict) and ("$Intermedia_" in k or "$Cause_" in k):
                text_parts.append(extract_dict_text(v))
            elif k.startswith("input") and isinstance(v, str):
                text_parts.append(v.strip())

        combined_text = "\n".join(filter(None, text_parts)).strip()
        if not combined_text:
            continue

        combined_text = expand_synonyms(combined_text)
        docs.append(Document(page_content=combined_text, metadata=metadata))
        loaded_count += 1

        # Update progress bar less frequently for performance
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
             progress_percentage = (i + 1) / total_files
             progress_bar.progress(progress_percentage, text=f"Processed files... ({i+1}/{total_files})")

    progress_bar.empty()
    st.write(f"Successfully loaded and preprocessed {loaded_count} documents.")
    if not docs:
         st.warning("No processable documents were found.")
    return docs

@st.cache_data(show_spinner="Chunking documents...")
def chunk_docs(_docs: List[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    """ Splits documents into smaller chunks. """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    new_docs = []
    if not _docs: return []

    st.write(f"Starting chunking for {len(_docs)} documents...")
    for doc in _docs:
        try:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                if chunk.strip():
                    new_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        except Exception as e:
            st.warning(f"Error chunking document from {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    st.write(f"Finished chunking. Generated {len(new_docs)} chunks.")
    return new_docs

# --- Vector Store Handling (Load or Build/Save) ---
@st.cache_resource(show_spinner=False) # Cache the final vectorstore object
def get_vectorstore(
    _docs_to_index: List[Document] | None,
    embedding_model_name: str,
    index_path: str,
    google_api_key: str # Required for Google embeddings
    ) -> FAISS | None:
    """
    Loads FAISS index from disk if available for the given Google model,
    otherwise builds and saves it using GoogleGenerativeAIEmbeddings.
    """
    if not _docs_to_index and not os.path.exists(index_path):
        st.error("Cannot build vector store: No documents provided and no existing index found.")
        return None
    if not google_api_key: # Check API key presence
        st.error("Google API Key is missing. Cannot initialize embeddings.")
        # This error should ideally be caught earlier, but double-check here.
        return None

    # Initialize Google embeddings
    try:
        st.write(f"Initializing Google embedding model: {embedding_model_name}")
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model=embedding_model_name
        )
    except Exception as e:
        st.error(f"Failed to initialize Google embeddings: {e}")
        st.error("Check your Google API Key and network connection.")
        return None

    # --- Load or Build Logic ---
    if os.path.exists(index_path):
        with st.spinner(f"Loading existing vector store from {index_path}..."):
            try:
                start_time = time.time()
                vectorstore = FAISS.load_local(
                    index_path,
                    embeddings, # Pass the Google embeddings object
                    allow_dangerous_deserialization=True # Needed for some embedding types
                )
                load_time = time.time() - start_time
                st.success(f"Vector store loaded successfully from {index_path} in {load_time:.2f}s.")
                return vectorstore
            except Exception as e:
                st.error(f"Error loading vector store from {index_path}: {e}")
                st.warning("This might happen if the index was built with different embeddings. Will attempt to rebuild.")
    else:
        st.info(f"No existing vector store found at {index_path}.")

    # --- Build and Save Logic ---
    if not _docs_to_index:
         st.error("Vector store needs to be built, but no documents were provided for indexing.")
         return None

    st.info(f"Building new vector store using {embedding_model_name}...")
    with st.spinner(f"Indexing {len(_docs_to_index)} document chunks... (This may take time using Google AI API)"):
        try:
            start_time = time.time()
            vectorstore = FAISS.from_documents(_docs_to_index, embeddings) # Use Google embeddings
            build_time = time.time() - start_time
            st.success(f"Vector store built successfully in {build_time:.2f}s.")

            # Save the index
            with st.spinner(f"Saving vector store to {index_path}..."):
                try:
                    vectorstore.save_local(index_path)
                    st.success(f"Vector store saved successfully to {index_path}.")
                except Exception as e_save:
                    st.error(f"Error saving vector store to {index_path}: {e_save}")
            return vectorstore

        except Exception as e_build:
            st.error(f"Error building vector store: {e_build}")
            st.error("Check your Google API quota, API key, and network connection.")
            return None

# --- RAG Chain Creation ---
@st.cache_resource(show_spinner="Initializing RAG Chain with Google LLM...")
def create_google_rag_chain(
    _vectorstore: FAISS | None,
    google_api_key: str, # Required for Google LLM
    llm_model_name: str = GOOGLE_LLM_MODEL
    ) -> RetrievalQA | None:
    """ Creates the RetrievalQA chain using Google Generative AI LLM. """
    if _vectorstore is None:
        st.error("Cannot create RAG chain: Vector store is not available.")
        return None
    if not google_api_key:
        st.error("Cannot create RAG chain: Google API Key is missing.")
        return None

    try:
        # Initialize Google LLM
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            google_api_key=google_api_key,
            temperature=0.5, # Adjust creativity
            convert_system_message_to_human=True # Often needed for chat models in RAG
        )
        st.write(f"Initialized Google LLM: {llm_model_name}")

        # Create Retriever
        retriever = _vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})

        # Create RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Suitable for models handling moderate context lengths
            retriever=retriever,
            return_source_documents=True,
        )
        st.write("RAG chain created successfully.")
        return qa_chain

    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        st.error("Check your Google API Key, model name, and network connection.")
        return None

# --- Streamlit App ---

st.title("🩺 MIMIC-IV Clinical Query Assistant (Google AI)")
st.markdown("""
Enter a clinical question. The system retrieves relevant MIMIC-IV info using Google AI embeddings
and generates an answer using Google's Gemini model.
**Note:** Demo only. Not for clinical use. Handle data ethically.
""")

# --- Initial API Key Checks ---
if not GOOGLE_API_KEY:
    st.error("Google API Key (GOOGLE_API_KEY) not found.")
    st.info("Please configure it via a `.env` file or Streamlit secrets.")
    st.stop()


# --- Initialization ---
vectorstore = None
qa_chain = None

with st.spinner("Initializing System... Loading data..."):
    raw_docs = load_mimic_finished(DATA_DIR) # Cached

if raw_docs:
    # Chunking is cached
    chunked_docs = chunk_docs(raw_docs)

    if chunked_docs:
        # Get Vector Store (Load or Build/Save) - uses Google AI Embeddings
        # Caching and spinners handled inside the function
        vectorstore = get_vectorstore(
            _docs_to_index=chunked_docs,
            embedding_model_name=GOOGLE_EMBEDDING_MODEL,
            index_path=VECTOR_STORE_PATH,
            google_api_key=GOOGLE_API_KEY
        )

        if vectorstore:
            # Create RAG chain using Google LLM
            # Caching and spinners handled inside the function
            qa_chain = create_google_rag_chain(
                _vectorstore=vectorstore,
                google_api_key=GOOGLE_API_KEY,
                llm_model_name=GOOGLE_LLM_MODEL
            )
        else:
            st.error("Failed to initialize vector store. Cannot proceed.")
    else:
        st.warning("No document chunks were created. Cannot build vector store.")
else:
    st.error("Failed to load documents. Please check data path and file integrity.")


# --- User Interaction ---
if qa_chain:
    st.divider()
    st.success("System Ready. Enter your query below.")
    query = st.text_input("Enter your clinical question:", key="query_input")

    if st.button("Get Answer", key="submit_button"):
        if query:
            with st.spinner("Retrieving documents and generating answer with Google Gemini..."):
                try:
                    start_query_time = time.time()
                    result = qa_chain.invoke({"query": query}) # Use invoke
                    end_query_time = time.time()

                    answer = result.get("result", "No answer generated.")
                    source_docs = result.get("source_documents", [])

                    st.subheader("Generated Answer")
                    st.markdown(answer)
                    st.info(f"Query processed in {end_query_time - start_query_time:.2f} seconds.")

                    st.subheader(f"Retrieved Documents (Top {len(source_docs)})")
                    if source_docs:
                        for i, doc in enumerate(source_docs):
                            with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                                st.write(f"**Content:**\n```\n{doc.page_content}\n```")
                    else:
                        st.write("No relevant documents were retrieved for this query.")

                except Exception as e:
                    st.error(f"An error occurred during query processing: {e}")
                    st.error("Please check your API keys, network connection, and API quotas (Google).")
        else:
            st.warning("Please enter a question.")

elif not raw_docs:
    st.error("Initialization failed: Could not load data.")
elif not vectorstore:
     st.error("Initialization failed: Could not load or build the vector store.")
elif not qa_chain:
     st.error("Initialization failed: Could not create the RAG chain. Check API keys/models.")
else:
    st.error("System initialization failed for an unknown reason.")


# --- Sidebar ---
st.sidebar.title("System Configuration")
st.sidebar.markdown("---")
st.sidebar.info(f"**Embedding Model (Google):**\n`{GOOGLE_EMBEDDING_MODEL}`")
st.sidebar.info(f"**LLM (Google):**\n`{GOOGLE_LLM_MODEL}`")
st.sidebar.info(f"**Vector Store Path:**\n`{VECTOR_STORE_PATH}`")
if os.path.exists(VECTOR_STORE_PATH):
    st.sidebar.success(f"Vector store index loaded.")
else:
    st.sidebar.warning(f"Vector store index not found. Will be built.")
st.sidebar.info(f"**Retrieved Docs (k):** {SEARCH_K}")
st.sidebar.info(f"**Chunk Size:** {CHUNK_SIZE}")

st.sidebar.markdown("---")
st.sidebar.title("Ethical Considerations")
st.sidebar.warning(
    "Using sensitive clinical data (MIMIC-IV). Ensure compliance with data use agreements, privacy, and ethical use. This tool is for educational/research purposes only."
)