from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict, PrivateAttr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import deduplicate_documents


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATION_MODEL = "google/flan-t5-base"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3


@dataclass
class RetrievalChunk:
    document: Document
    score: float


class LocalSeq2SeqLLM(LLM):
    """Minimal LangChain LLM wrapper around a local seq2seq Transformers model."""

    model_name: str
    max_new_tokens: int = 256
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_name: str, max_new_tokens: int = 256) -> None:
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @property
    def _llm_type(self) -> str:
        return "local_seq2seq"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=False,
            temperature=0.0,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        if stop:
            for stop_token in stop:
                if stop_token in text:
                    text = text.split(stop_token)[0]
        return text.strip()


class RAGPipeline:
    """Document QA pipeline built with LangChain, FAISS, and Hugging Face."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        generation_model_name: str = DEFAULT_GENERATION_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embeddings = self._build_embeddings()
        self.llm = self._build_llm()
        self.vectorstore: FAISS | None = None
        self.qa_chain: RetrievalQA | None = None
        self.document_path: Path | None = None
        self.page_count = 0
        self.chunk_count = 0

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_llm(self) -> LocalSeq2SeqLLM:
        return LocalSeq2SeqLLM(
            model_name=self.generation_model_name,
            max_new_tokens=256,
        )

    def _build_prompt(self) -> PromptTemplate:
        template = (
            "You are answering questions about a PDF document.\n"
            "Use only the retrieved context below.\n"
            'If the answer is not present, say "I do not know based on the provided document.".\n\n'
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

    def load_and_split_documents(self, pdf_path: str | Path) -> list[Document]:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        self.page_count = len(documents)

        # Recursive splitting preserves paragraphs and sentences before falling back to words.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        self.chunk_count = len(chunks)
        return chunks

    def ingest_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        self.document_path = Path(pdf_path)
        chunks = self.load_and_split_documents(pdf_path)

        if not chunks:
            raise ValueError("The provided PDF did not produce any text chunks.")

        # Build a fresh FAISS index for the selected PDF.
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self._build_prompt()},
        )

        return {
            "document_path": str(self.document_path),
            "num_pages": self.page_count,
            "num_chunks": self.chunk_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model_name,
            "generation_model": self.generation_model_name,
        }

    def retrieve_chunks(self, question: str) -> list[RetrievalChunk]:
        if self.vectorstore is None:
            raise ValueError("No vector store available. Ingest a PDF first.")

        results = self.vectorstore.similarity_search_with_score(
            question,
            k=self.top_k,
        )
        return [
            RetrievalChunk(document=doc, score=1.0 / (1.0 + float(score)))
            for doc, score in results
        ]

    def answer_question(self, question: str) -> dict[str, Any]:
        if self.qa_chain is None:
            raise ValueError("QA chain is not initialized. Ingest a PDF first.")

        retrieved_chunks = self.retrieve_chunks(question)
        response = self.qa_chain.invoke({"query": question})
        source_documents = deduplicate_documents(response.get("source_documents", []))

        score_lookup = {
            (item.document.page_content, str(item.document.metadata.get("page", ""))): item.score
            for item in retrieved_chunks
        }

        return {
            "answer": response["result"].strip(),
            "retrieved_chunks": [self._serialize_chunk(item.document, item.score) for item in retrieved_chunks],
            "source_documents": [
                self._serialize_chunk(
                    doc,
                    score_lookup.get((doc.page_content, str(doc.metadata.get("page", "")))),
                )
                for doc in source_documents
            ],
        }

    @staticmethod
    def _serialize_chunk(doc: Document, score: float | None) -> dict[str, Any]:
        return {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
