import streamlit as st

import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Document
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.utils import Secret

template = """
Forget everything you read from any previous prompts.
Language: Dutch. 
Gebruik de volgende informatie, en alleen de volgende informatie om de vraag te beantwoorden:

Context:
{% for document in documents %}
    {{ document.content }}
    {{ document.meta}}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

generator = OpenAIGenerator(model="gpt-3.5-turbo")

secret = Secret.from_token(os.getenv("MONGO_CONNECTION_STRING"))

document_store = MongoDBAtlasDocumentStore(
        mongo_connection_string=secret,
        database_name="clusterTest",
        collection_name="collectionTest",
        vector_search_index="vector_index",
    )

if "basic_rag_pipeline" not in st.session_state:
    retriever = MongoDBAtlasEmbeddingRetriever(document_store=document_store)
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)
    basic_rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")
    basic_rag_pipeline.connect("llm.replies", "answer_builder.replies")
    basic_rag_pipeline.connect("llm.meta", "answer_builder.meta")
    basic_rag_pipeline.connect("retriever", "answer_builder.documents")
    st.session_state["basic_rag_pipeline"] = basic_rag_pipeline

query = st.text_input("Ask your question")

if query:
    results = st.session_state["basic_rag_pipeline"].run(
    data={"text_embedder": {"text": query}, "retriever": {"top_k": 5}, 
          "prompt_builder": {"question": query}, "answer_builder": {"query": query}})
    for answer in results["answer_builder"]["answers"]:
        st.write(answer.data)
        st.write(answer.documents)