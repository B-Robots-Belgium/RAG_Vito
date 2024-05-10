import streamlit as st
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from collections import Counter

if 'extractive_qa_pipeline' not in st.session_state:
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=st.session_state['small_model']), name="query_embedder_search")
    rag_pipeline.add_component(instance=MongoDBAtlasEmbeddingRetriever(document_store=st.session_state['document_store']), name="retriever_search")
    rag_pipeline.connect("query_embedder_search", "retriever_search.query_embedding")
    st.session_state['extractive_qa_pipeline'] = rag_pipeline

query = st.text_input("Test to search for or compare.")

if query:
    results = st.session_state['extractive_qa_pipeline'].run(
    data={"query_embedder_search": {"text": query}, "retriever_search": {"top_k": 5}}
    )
    results_cleaned = []
    tracking_classifications = []
    for index in range(len(results["retriever_search"]["documents"])):
        results_cleaned.append({
            f'{results["retriever_search"]["documents"][index].meta["article_id"]}': f'{results["retriever_search"]["documents"][index].meta["article_id"]}',
            'Linked Classifications': f'{results["retriever_search"]["documents"][index].meta["sources"][0]}',
            'Document': f'{results["retriever_search"]["documents"][index].content}',
            'Confidence': f'{results["retriever_search"]["documents"][index].score}'
        })
        tracking_classifications.append(results["retriever_search"]["documents"][index].meta["sources"][0])

    st.write(results_cleaned)
    flattened_list = [item for sublist in tracking_classifications for item in sublist]
    counters = Counter(flattened_list)
    max_count = max(counters.values())
    max_occurrence_items = {item for item, count in counters.items() if count == max_count}
    st.write(list(max_occurrence_items))
