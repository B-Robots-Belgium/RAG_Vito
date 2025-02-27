import psycopg2
import openai
from utils import clean_html
from .vito_classes import VitoArticle
from db_actions import get_top_similar_items
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import PGVector

def UploadOpenAI(text: str, client):
    response = client.chat.completions.create(
        model = "gpt-4o",
        response_format= {"type": "json_object"},
        messages = [
            {"role": "system", "content": "You are generating metadata for the following article. Always return the found information in JSON format"},
            {"role": "system", "content": "You will always return the information in the following JSON format, with Keyword being a general filler and the 'xxx' being the extraxted keyword:"},
            {"role": "system", "content": "{'metadata': [ {'Keyword': 'xxx'}, {'Keyword': 'xxx'} ]}"},
            {"role": "system", "content": "The metadata should be keywords that describe the article and are found within the article. Preferably these are keywords that can be used for clustering techniques. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "Extract only the 10 most relevant keywords from the following article. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "All keywords returned should be in Dutch. The text you will receive is also in Dutch."},
            {"role": "system", "content": "There should be no specific measurements. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "The following message holds the text from which to generate the the metadata. Always return the generated metadata in JSON format"},
            {"role": "user", "content": text}
        ])
    return response

def classify_with_langchain(article: VitoArticle, connection_string: str, collection_name: str = "vito_articles"):
    """
    Classifies an article using retrieval (stored in PostgreSQL / pgvector) and GPT when necessary.

    :param article: A VitoArticle instance.
    :param connection_string: Your PostgreSQL connection string, e.g. "postgresql+psycopg2://user:pass@host:5432/dbname"
    :param collection_name: The name of the pgvector collection/table where embeddings are stored.
    :return: The classification result returned by the RetrievalQA chain.
    """
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    retriever = vectorstore.as_retriever()

    llm = OpenAI(model_name="gpt-4o")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",        
        retriever=retriever
    )

    response = qa.run(f"Classify the following article: {clean_html(article.inhoud)}")
    return response


def process_article(json_file, openai_client, faiss_index):
    """Processes a new article: extracts keywords, updates embedding, and classifies."""
    article = VitoArticle(json_file)
    article.extract_keywords()
    article.update_embedding(openai_client)
    
    # Perform retrieval to find similar articles
    similar_articles = get_top_similar_items(psycopg2.connect(
        database="postgres",
        user="postgres",
        password="yourpassword"), article.embedding, 5)
    
    if not similar_articles or max([sim[2] for sim in similar_articles]) < 0.7:
        classification = classify_with_langchain(article, faiss_index)
    else:
        classification = [sim[3] for sim in similar_articles]
    
    return article, classification