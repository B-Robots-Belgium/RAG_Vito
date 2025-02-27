from vito_utils.utils import clean_html
from vito_utils.vito_classes import VitoArticle
from vito_utils.vito_classification import assign_labels_a_star
from vito_utils.keyword_utils import extract_keybert_keywords
from openai import OpenAI
import os
import json
import psycopg2



if __name__ == "__main__": 
    openaiClient = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization= os.getenv("OPENAPI_ORG")
    )

    PgVectorConn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="root",
        port="5432"
    )

    

    PgVectorConn.close()

    # list_result_labels = []

    # article = VitoArticle("C:/Users/EhranLenaerts/Documents/RAG_B_Robots/RAG_Vito/data/Stoffen/Aard/Afvalstoffen/Bijzondere afvalstoffen/gebruikte PCB's/44412.json")
    # if article.inhoud == "<div>\r\n     <div>[...]</div>\r\n   </div>":
    #     print('----------------------------------------------------------------------------------------------------------------')
    #     print("Skipping due to invalid HTML/article")
    # else:
    #     # try:
    #     article.inhoud = clean_html(article.inhoud)
    #     article.embedding = article.get_embedding(openaiClient)
    #     article.extract_keywords(top_n=10)
    #     print(article.weighted_keywords)
    #     result_labels, assigned_labels = assign_labels_a_star(testArticle=article, PgVectorConn=PgVectorConn)
    #     list_result_labels.append({
    #         "Artikel": article.artikel_id, 
    #         "Artikel_name": article.artikel, 
    #         "List_result_labels": result_labels, 
    #         "correct_labels": article.labels})
    #     print('----------------------------------------------------------------------------------------------------------------')
    #     print(f"Article: {article.artikel_id}, Labels: {result_labels}, correct labels: {article.labels}")
    #     # except Exception as e:
    #     #     print(f"Article: {article.artikel_id}, {e}")


