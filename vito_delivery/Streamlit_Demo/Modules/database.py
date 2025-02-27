# Modules/database.py

import psycopg2

def get_top_similar_items(conn, query_vector, top_n=5):
    """
    Retrieve the top N items most similar to the query vector using cosine similarity.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH target AS (
                    SELECT %s::VECTOR(3072) AS vector
                )
                SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                FROM vito_article, target
                ORDER BY similarity ASC
                LIMIT %s;
            """, (query_vector, top_n))
            results = cur.fetchall()
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        results = []
    return results

def get_all_items(conn):
    """
    Retrieve all items from the database.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT artikel_id, chunk_id, keywords, labels FROM vito_articles;
            """)
            results = cur.fetchall()
    except Exception as e:
        print(f"Error retrieving items: {e}")
        results = []
    return results

def query_semantically_alike_items(conn, query_vector, top_label, top_n=0):
    """
    Retrieve items similar to the query vector within a specific label.
    """
    try:
        top_label = top_label.replace('\\', ',')

        if " " in top_label.split(',')[-1]:
            top_label = top_label.split(',')[0] + ',' + '%' + top_label.split(',')[-1] + '%'

        if top_n == 0:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH target AS (
                        SELECT %s::VECTOR(3072) AS vector
                    )
                    SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                    FROM vito_article, target
                    WHERE labels LIKE %s
                    ORDER BY similarity ASC;
                """, (query_vector, "%" + top_label + "%"))
                results = cur.fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH target AS (
                        SELECT %s::VECTOR(3072) AS vector
                    )
                    SELECT artikel_id, chunk_id, weighted_keywords, labels, (embedding <=> target.vector) AS similarity
                    FROM vito_article, target
                    WHERE labels LIKE %s
                    ORDER BY similarity ASC
                    LIMIT %s;
                """, (query_vector, "%" + top_label + "%", top_n))
                results = cur.fetchall()
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        results = []
    return results