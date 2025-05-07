# 🧬 ClinicalRAG: Gemini-Powered Clinical Document Search with MIMIC-III

**ClinicalRAG** is a Retrieval-Augmented Generation (RAG) system that leverages **Gemini Pro** to enable natural language querying over the **MIMIC-III** clinical dataset. It empowers healthcare practitioners and researchers to interactively search and reason over real-world ICU records using state-of-the-art large language models.

---

## 🔍 Overview

This project integrates:

- **MIMIC-III Dataset**: A de-identified database of over 60,000 ICU admissions.
- **Gemini Pro**: A multimodal large language model for contextual reasoning and response generation.
- **RAG Architecture**: Combines semantic search via vector embeddings with generative answering using Gemini.

---

## 📁 Project Structure

```

clinicalrag/
│
├── data/                   # Preprocessed MIMIC-III clinical notes
├── embeddings/             # FAISS vector index and metadata
├── app.py                  # Streamlit app for the user interface
├── gemini\_rag.py           # Core RAG logic with Gemini API integration
├── utils.py                # Helper functions and utilities
├── requirements.txt        # Python dependencies
└── README.md               # This documentation file

````

---

## 🚀 Features

- Upload or use built-in MIMIC-III clinical notes
- FAISS-powered semantic search on clinical note embeddings
- Gemini Pro generates detailed, contextual responses to medical queries
- Intuitive Streamlit web interface for real-time interaction

---

## 🧠 How It Works

1. **Embedding**: Clinical notes are embedded using Google’s text embedding model.
2. **Indexing**: FAISS is used to index and retrieve the most relevant note chunks.
3. **RAG Pipeline**: Retrieved chunks and the user’s query are passed to Gemini for final response generation.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clinicalrag.git
cd clinicalrag
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Gemini API Key

Set your Gemini API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

> On Windows (CMD):

```cmd
set GOOGLE_API_KEY=your-api-key-here
```

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Example Queries

Here are some examples of natural language queries you can ask:

* *What medications were given during the ICU stay?*
* *Did the patient show signs of pneumonia?*
* *What was the discharge diagnosis?*

---

## 📊 Dataset Information

* **Source**: [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)
* **Type**: De-identified ICU data (notes, diagnoses, procedures, etc.)
* **License**: PhysioNet Credentialed Health Data Use Agreement

---

## 📦 Dependencies

* `streamlit`
* `faiss-cpu`
* `openai`
* `google-generativeai`
* `pandas`
* `numpy`
* `tqdm`

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Muhammad Bin Imran**
FAST National University of Computer and Emerging Sciences
📧 [muhammadbinimran1000@gmail.com](mailto:muhammadbinimran1000@gmail.com)

---

> ⚠️ **Disclaimer**: This tool is intended for educational and research purposes only. It is not approved for clinical use or medical decision-making.

```
