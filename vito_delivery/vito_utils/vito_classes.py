import pandas as pd
import json
from datetime import date, datetime
from .utils import clean_html
from .local_metadata.vito_enum import Thema, Type, Toepassingsgebied
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
import requests

def split_text_to_fit_token_limit(text: str, max_tokens: int) -> list:
    """Splits the text into chunks that fit within the model's max token limit."""
    max_chars = int(max_tokens * 2)  # Estimate max characters based on token limit
    print(max_chars)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

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
    
    def extract_keywords(self, top_n=10, use_mmr: bool = True, stop_words=None, diversity: int = 0.5):
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        sentence_model = SentenceTransformer(model_name)
        kw_model = KeyBERT(sentence_model)
        """Extracts keywords using KeyBERT."""
        self.keywords = kw_model.extract_keywords(
            clean_html(self.inhoud),
            top_n=top_n,
            keyphrase_ngram_range=(1, 2),
            stop_words=stop_words,
            use_mmr=use_mmr,
            diversity=diversity
        )
        self.weighted_keywords = {kw[0]: kw[1] for kw in self.keywords}
    
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
                if  len(self.embedding) == 2:
                    print("Chunking item: " + self.artikel_id)
                    for chunk_id, embedding in enumerate(self.embedding):
                        # Insert or update each chunk separately
                        
                        cur.execute("""
                        INSERT INTO vito_article (artikel_id, chunk_id, url, labels, keywords, embedding, weighted_keywords, artikel_title)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (artikel_id, labels, chunk_id, artikel_title) DO UPDATE 
                        SET url = EXCLUDED.url,
                            labels = EXCLUDED.labels,
                            keywords = EXCLUDED.keywords,
                            embedding = EXCLUDED.embedding,
                            weighted_keywords = EXCLUDED.weighted_keywords,
                            artikel_title = EXCLUDED.artikel_title
                        """, (self.artikel_id, chunk_id, self.url, self.labels, "{" + ",".join(f'"{kw[0]}"' for kw in self.keywords) + "}", embedding, json.dumps(self.weighted_keywords), self.artikel))
                        PgVectorConn.commit()
                else:
                    cur.execute("""
                        INSERT INTO vito_article (artikel_id, chunk_id, url, labels, keywords, embedding, weighted_keywords, artikel_title)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (artikel_id, labels, chunk_id, artikel_title) DO UPDATE 
                        SET url = EXCLUDED.url,
                            labels = EXCLUDED.labels,
                            keywords = EXCLUDED.keywords,
                            embedding = EXCLUDED.embedding,
                            weighted_keywords = EXCLUDED.weighted_keywords,
                            artikel_title = EXCLUDED.artikel_title
                        """, (self.artikel_id, '0', self.url, self.labels, "{" + ",".join(f'"{kw[0]}"' for kw in self.keywords) + "}", self.embedding, json.dumps(self.weighted_keywords), self.artikel))
                    PgVectorConn.commit()
        except Exception as e:
            PgVectorConn.rollback()
            print(e)
        finally:
            cur.close()
            
    
    def generate_embeddings_with_error_handling(self, openaiClient, max_tokens: int = 8192, model: str ="text-embedding-3-large"):
        """Generates embeddings for text, dynamically handling token limits."""
        text = self.inhoud.replace("\n", " ").replace("\t", "")
        try:
            """Creates an embedding for the article using the GPT API."""
            self.embedding = openaiClient.embeddings.create(input = [text], model=model).data[0].embedding
        
        except Exception as e:
            print(f"Original id: {self.artikel_id}")
            error_message = str(e)
            print(error_message.split('- ')[1].split('requested ')[1].split(' tokens')[0])
            if 'maximum context length' in error_message:
                print("Token limit exceeded. Adjusting the text into smaller chunks...")
                # Split the text into smaller chunks based on estimated tokens
                text_chunks = split_text_to_fit_token_limit(text, max_tokens)
                embeddings = []
                for chunk in text_chunks:
                    text_chunk = chunk.replace("\n", "").replace("\t", "")
                    try:
                        response = openaiClient.embeddings.create(input = [text_chunk], model=model).data[0].embedding
                    except Exception as e:
                        print(e)
                    embeddings.append(response)
            self.embedding = embeddings

class VitoBoek():
    def __init__(self, boek_data: dict):
        list_items = dict(map(lambda item: (item['ref'], item['href']), boek_data['metadata']))
        self.id = boek_data['id']
        self.toepassingsgebied = Toepassingsgebied(int(list_items['TOEPASSINGSGEBIED'].split('/')[-1]))
        self.type = Type(int(list_items['TYPE'].split('/')[-1]))
        self.thema = Thema(int(list_items['THEMA'].split('/')[-1]))
        self.versies_url = list_items['VERSIES']
    
    def get_latest_versie(self):
        today_date = date.today()
        versies = requests.get(self.versies_url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()
        self.versie = min(
            versies, key=lambda item: abs(
                datetime.strptime(item['datumVanKracht'], "%Y-%m-%d").date() - today_date
            )
        )
    
    def get_boek_info(self):
        self.get_latest_versie()
        boek_item = requests.get(self.versie['metadata'][0]['href'], auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()
        self.samenvatting = boek_item['samenvatting']
        self.titel = boek_item['titel']
        return {
            "samenvatting": self.samenvatting,
            "titel": self.titel
        }
    