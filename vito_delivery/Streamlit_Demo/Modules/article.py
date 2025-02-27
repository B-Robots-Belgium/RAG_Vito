# Modules/article.py

import json
import os

# Class to handle article objects
class VitoArticle():
    def __init__(self, json_file: str, keywords: list = [], embedding: list = []):
        with open(json_file) as f:
            json_article = json.load(f)
            self.inhoud = json_article["inhoud"]
            self.labels = json_article["metadata"][0]
            self.url = json_article["metadata"][1]
            self.artikel_id = json_article["artikel id"]
            self.artikel = json_article["artikel"]
            try:
                self.embedding = json_article["embedding"]
            except:
                self.embedding = []
            try:
                self.keywords = json_article["metadata"][2]
                self.weighted_keywords = {keyword.lower(): 1.0 for keyword in self.keywords}
            except: 
                self.keywords = []
                self.weighted_keywords = []
    
    def add_keywords(self, keywords: list):
        self.keywords = keywords
        self.weighted_keywords = {keyword: 1.0 for keyword in tuple(self.keywords)}
    
    def save_to_json(self, json_file: str):
        data = {
            "inhoud": self.inhoud,
            "metadata": [
                self.labels,
                self.url,
                self.keywords
            ],
            "artikel id": self.artikel_id,
            "artikel": self.artikel,
            "embedding": self.embedding
        }
        with open(json_file, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load_from_json(cls, json_file: str):
        return cls(json_file)

    def get_embedding(self, openaiClient, model="text-embedding-3-large"):
        """Creates an embedding for the article using the GPT API."""
        text = self.inhoud.replace("\n", " ")
        return openaiClient.embeddings.create(input = [text], model=model).data[0].embedding

    def store_embedding_in_pgvector(self, PgVectorConn):
        try:
            with PgVectorConn.cursor() as cur:
                # Insert or update the article and its embedding
                cur.execute("""
                INSERT INTO vito_articles (artikel_id, url, labels, keywords, inhoud, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (artikel_id) DO UPDATE 
                SET url = EXCLUDED.url,
                    labels = EXCLUDED.labels,
                    keywords = EXCLUDED.keywords,
                    inhoud = EXCLUDED.inhoud,
                    embedding = EXCLUDED.embedding
                """, (self.artikel_id, self.url, self.labels, self.keywords, self.inhoud.replace("\n", " ").replace("\t", " "), self.embedding))
                PgVectorConn.commit()
        except Exception as e:
            PgVectorConn.rollback()
            print(e)
        finally:
            cur.close()