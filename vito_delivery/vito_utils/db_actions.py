from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List

class PgVectorRetriever(VectorStoreRetriever):

    def __init__(self, conn, embedder, top_k=5):
        pass


def get_top_similar_items(conn, query_vector, top_n=5):
    """
    Retrieve the top N items most similar to the query vector using cosine similarity.

    :param conn: PostgreSQL connection object
    :param query_vector: The vector to compare against stored vectors
    :param top_n: Number of top similar items to retrieve
    :return: List of tuples containing (article_id, chunk_id, similarity_score)
    """
    try:
        with conn.cursor() as cur:
            # Perform the similarity search using cosine similarity
            cur.execute("""
                with target AS (
	                SELECT %s::VECTOR(3072) AS vector
                )
                SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                FROM vito_article, target
                ORDER BY similarity ASC
                LIMIT %s;
            """, (f'{query_vector}', top_n))

            # Fetch the results
            results = cur.fetchall()

    except Exception as e:
        print(f"Error performing similarity search: {e}")
        results = []

    return results

def get_all_items(conn):
    """
    Retrieve the top N items most similar to the query vector using cosine similarity.

    :param conn: PostgreSQL connection object
    :param query_vector: The vector to compare against stored vectors
    :param top_n: Number of top similar items to retrieve
    :return: List of tuples containing (article_id, chunk_id, similarity_score)
    """
    # query_vector_np = np.array(query_vector).tolist()  # Convert vector to list for SQL compatibility

    try:
        with conn.cursor() as cur:
            # Perform the similarity search using cosine similarity
            cur.execute("""
                SELECT artikel_id, chunk_id, keywords, labels FROM vito_article;
            """)

            # Fetch the results
            results = cur.fetchall()

    except Exception as e:
        print(f"Error performing similarity search: {e}")
        results = []

    return results

def query_sematically_alike_items(conn, query_vector, topLabel, top_n=0):
    """
    Retrieve the top N items most similar to the query vector using cosine similarity.

    :param conn: PostgreSQL connection object
    :param query_vector: The vector to compare against stored vectors
    :param top_n: Number of top similar items to retrieve
    :return: List of tuples containing (article_id, chunk_id, similarity_score)
    """
    try:
        topLabel = topLabel.replace('\\', ',')
        
        if " " in topLabel.split(',')[-1]:
            topLabel =  topLabel.split(',')[0] + ',' + '%' +topLabel.split(',')[-1] + '%'
        
        if top_n == 0:
            with conn.cursor() as cur:
            # Perform the similarity search using cosine similarity
                cur.execute("""
                    with target AS (
                        SELECT %s::VECTOR(3072) AS vector
                    )
                    SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                    FROM vito_article, target
                    WHERE labels LIKE %s
                    ORDER BY similarity ASC;
                """, (f'{query_vector}', "%" + topLabel + "%"))

                # Fetch the results
                results = cur.fetchall()
        else:
            with conn.cursor() as cur:
            # Perform the similarity search using cosine similarity
                cur.execute("""
                    with target AS (
                        SELECT %s::VECTOR(3072) AS vector
                    )
                    SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                    FROM vito_article, target
                    WHERE labels LIKE %s
                    ORDER BY similarity ASC
                    LIMIT %s;
                """, (f'{query_vector}', "%" + topLabel + "%", top_n))

                # Fetch the results
                results = cur.fetchall()
        

    except Exception as e:
        print(f"Error performing similarity search: {e}")
        results = []

    return results