# Paracetamol Document QA with RAG

This project is a complete Retrieval-Augmented Generation (RAG) application for answering questions about a PDF document on Paracetamol. It follows the required architecture using LangChain, Hugging Face embeddings, FAISS, and Streamlit.

## Project Structure

```text
rag_paracetamol_project/
|-- app.py
|-- rag_pipeline.py
|-- utils.py
|-- requirements.txt
|-- README.md
`-- data/
    `-- paracetamol.pdf
```

## Architecture Flow

```text
User Question
    |
Streamlit UI
    |
Retriever (FAISS Vector DB)
    |
Top-k relevant chunks
    |
LLM generates answer using context
    |
Return answer + source chunks
```

## Chunking Strategy

The PDF is loaded with `PyPDFLoader` and split with `RecursiveCharacterTextSplitter` using:

- `chunk_size=600`
- `chunk_overlap=80`

These values are intentionally chosen within the required range:

- A chunk size around 600 characters keeps meaningful medical content together, such as usage notes, warnings, or dosage details.
- An overlap of 80 characters reduces boundary loss when a sentence spans two neighboring chunks.
- This balance improves retrieval precision without stripping away too much context.

## Embeddings

The project uses `HuggingFaceEmbeddings` with the model `sentence-transformers/all-MiniLM-L6-v2`.

Embeddings are vector representations of text. They map chunks and questions into the same semantic space so the retriever can compare meaning instead of relying only on exact word overlap.

Embeddings are essential in RAG because semantic search depends on them. Once both the query and the chunks are embedded, FAISS can quickly find the chunks whose meanings are closest to the user question.

## Retrieval Process

1. Load the PDF with `PyPDFLoader`.
2. Split the document into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. Convert each chunk into an embedding using `all-MiniLM-L6-v2`.
4. Store those vectors in a FAISS index.
5. Embed the user question.
6. Retrieve the top 3 most relevant chunks from FAISS.
7. Pass those chunks to a LangChain `RetrievalQA` chain.
8. Generate an answer constrained to the retrieved context.
9. Return the final answer and the source chunks used.

## Streamlit Features

The app includes:

- PDF uploader
- question input box
- Ask Question button
- loading spinner
- final answer display
- retrieved chunk display
- source chunk display
- similarity score display for each retrieved chunk

## Implementation Notes

- `app.py` contains the Streamlit interface and avoids re-indexing the same PDF on every question by caching ingestion state.
- `rag_pipeline.py` contains the end-to-end RAG pipeline using LangChain, Hugging Face embeddings, FAISS, and a LangChain `RetrievalQA` chain.
- `utils.py` contains helper functions for saving uploads, formatting chunks, and tracking the active document configuration.
- The default generation model is `google/flan-t5-base`, loaded locally through `transformers`.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Default Document

If you do not upload a file, the app will use:

```text
data/paracetamol.pdf
```

## Error Handling

The app handles the common failure cases gracefully:

- missing PDF
- empty question
- PDF with no extractable text
- model or dependency loading failures surfaced through Streamlit error output
