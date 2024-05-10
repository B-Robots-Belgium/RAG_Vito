import streamlit as st

import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Document
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders import PromptBuilder

template = """
Gebruik de volgende informatie, en alleen de volgende informatie:
Always forget all the information from the previous prompt.
Language: Dutch. 
Deze prompt is een classificatie taak.
Geef de meest correct classificaties van de lijst van de verschillende documenten, deze classificaties zitten in de meta lijst, achter sources.
Gebruik deze classificaties om dan de beste classificaties te kiezen voor deze question, dit mogen er maximum 3 zijn.
Geef ook een korte samenvatting van de question en waarom de volgende classificaties hier best bij horen.
Antwoord moet alijd in JSON formaat zijn in de stijl van:
Categorie: <Classification>
Samenvatting: <Summary>
De volgende context zijn de documenten waaruit je de informatie moet halen.
Context:
{% for document in documents %}
    {{ document.content }}
    {{ document.meta}}
{% endfor %}
De volgende question is het document dat je moet classificeren.
Question: {{question}}
"""

prompt_builder = PromptBuilder(template=template)

generator = OpenAIGenerator(model="gpt-3.5-turbo")

if "basic_rag_clas" not in st.session_state:
    retriever = MongoDBAtlasEmbeddingRetriever(document_store=st.session_state['document_store'])
    text_embedder = SentenceTransformersTextEmbedder(model=st.session_state['small_model'])
    basic_rag_clas = Pipeline()
    basic_rag_clas.add_component("text_embedder", text_embedder)
    basic_rag_clas.add_component("retriever", retriever)
    basic_rag_clas.add_component("prompt_builder", prompt_builder)
    basic_rag_clas.add_component("llm", generator)
    basic_rag_clas.add_component(instance=AnswerBuilder(), name="answer_builder")
    basic_rag_clas.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_clas.connect("retriever.documents", "prompt_builder.documents")
    basic_rag_clas.connect("prompt_builder", "llm")
    basic_rag_clas.connect("llm.replies", "answer_builder.replies")
    basic_rag_clas.connect("llm.meta", "answer_builder.meta")
    basic_rag_clas.connect("retriever", "answer_builder.documents")
    st.session_state["basic_rag_clas"] = basic_rag_clas

query = st.text_input("Ask your question")

if query:
    results = st.session_state["basic_rag_clas"].run(
    data={"text_embedder": {"text": query}, "retriever": {"top_k": 5}, 
          "prompt_builder": {"question": query}, "answer_builder": {"query": query}})
    for answer in results["answer_builder"]["answers"]:
        st.write(answer.data)
    