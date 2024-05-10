import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from datasets import load_dataset
import pandas as pd
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.utils import Secret
from haystack import Pipeline
from haystack.components.writers import DocumentWriter

secret = Secret.from_token(os.getenv("MONGO_CONNECTION_STRING"))

if 'openai_key' not in st.session_state:
    st.session_state["openai_key"] = os.getenv("OPENAI_API_KEY")

if 'small_model' not in st.session_state:
    small_model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    document_embedder = SentenceTransformersDocumentEmbedder(model=small_model)
    st.session_state['document_embedder'] = document_embedder
    text_embedder = SentenceTransformersTextEmbedder(model=small_model)
    st.session_state['text_embedder'] = text_embedder
    st.session_state['small_model'] = small_model

if 'document_store' not in st.session_state:
    document_store = MongoDBAtlasDocumentStore(
        mongo_connection_string=secret,
        database_name="vitoData",
        collection_name="vitoMiniLM",
        vector_search_index="vito_index",
    )
    st.session_state['document_store'] = document_store
    
if 'retriever' not in st.session_state:
    retriever = MongoDBAtlasEmbeddingRetriever(document_store=document_store)
    st.session_state['retriever'] = retriever

if 'reader' not in st.session_state:
    reader = ExtractiveReader()
    reader.warm_up()
    st.session_state['reader'] = reader

st.title("RAG Demo")

st.image("B_ROBOTS2019.webp")
