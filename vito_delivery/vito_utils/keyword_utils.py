from collections import defaultdict
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from .db_actions import get_top_similar_items

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#paraphrase-multilingual-mpnet-base-v2
#all-mpnet-base-v2
sentence_model = SentenceTransformer(model_name)
kw_model = KeyBERT(model_name)

def print_overlapping_keywords(stage, path, test_keywords, articles_df):
    """
    Prints information about overlapping keywords between test keywords and article keywords 
    for each article in the provided DataFrame, including article details and matching keywords.

    Args:
        stage (int): The current stage or step in the process.
        path (list): A list representing the hierarchical path or breadcrumb trail.
        test_keywords (dict): A dictionary of test article keywords with their associated weights.
        articles_df (pd.DataFrame): A DataFrame containing article details and their keywords.

    Prints:
        Details of each article including article ID, labels, semantic similarity, combined score,
        test and article keywords, and any overlapping keywords found.
    """
    logging.info("---------------------------------------------------------------------------")
    logging.info(f"\nStage {stage}, Level: {' > '.join(path) if path else 'Initial items'}")
    for idx, row in articles_df.iterrows():
        article_keywords = row['weighted_keywords']
        logging.info(f"\nArticle ID: {row['artikel']}")
        logging.info(f"Labels: {row['labels']}")
        logging.info(f"Semantic similarity: {row['weight']}")
        logging.info(f"Combined score: {row['combined_score']}\n")
        logging.info("Test Article Keywords: %s", test_keywords)
        logging.info("Article Keywords: %s", list(article_keywords.keys()))

        overlapping_keywords = []

        # Convert keywords to lowercase for case-insensitive comparison
        test_keywords_lower = {list(kw.keys())[0].lower():list(kw.values())[0] for kw in test_keywords}
        article_keywords_lower = {kw.lower(): kw for kw in article_keywords.keys()}

        # Check for substring overlaps between test and article keywords
        for test_kw_lower, test_kw_original in test_keywords_lower.items():
            for article_kw_lower, article_kw_original in article_keywords_lower.items():
                if test_kw_lower in article_kw_lower or article_kw_lower in test_kw_lower:
                    # Add the overlapping keywords and their weights
                    article_weight = article_keywords[article_kw_original]
                    overlapping_keywords.append({
                        'test_keyword': test_kw_original,
                        'article_keyword': article_kw_original,
                        'article_weight': article_weight
                    })
                    # Stop after first match for this test_kw_lower
                    break  

        if overlapping_keywords:
            logging.info("Overlapping Keywords:")
            for overlap in overlapping_keywords:
                logging.info(f"Test Keyword: '{overlap['test_keyword']}', "
                      f"Article Keyword: '{overlap['article_keyword']}', "
                      f"Article Keyword Weight: {overlap['article_weight']}")
        else:
            logging.info("No overlapping keywords.")

def calculate_weighted_keyword_similarity(test_keywords, article_keywords, similarity = 0.0):
    """
    Calculate the total weight of overlapping keywords between the test article's keywords and the database article's keywords.
    A keyword is considered overlapping if it is a substring of another keyword (case-insensitive).

    Args:
        test_keywords (dict): Dictionary of test article keywords with weights.
        article_keywords (dict): Dictionary of database article keywords with variable weights.

    Returns:
        float: Total similarity score based on overlapping keywords.
    """

    # Convert keywords to lowercase for case-insensitive comparison
    test_keywords_lower = {kw.lower(): kw for kw in test_keywords[0].keys()}
    article_keywords_lower = {kw.lower(): kw for kw in article_keywords.keys()}

    # For each test keyword, check if it overlaps with any article keyword
    for test_kw_lower in test_keywords_lower.keys():
        for article_kw_lower, article_kw_original in article_keywords_lower.items():
            if test_kw_lower in article_kw_lower or article_kw_lower in test_kw_lower:
                # Add the weight of the matching article keyword
                article_weight = article_keywords[article_kw_original]
                similarity += article_weight
                # Stop after first match for this test_kw_lower
                break  
    
    # Return the total similarity score
    return similarity 

def build_keyword_label_mapping(data):
    keyword_label_map = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        keywords = item[2]
        labels = item[3].replace('"', '').replace('{', '').replace('}', '').split(',')
        
        # Increment the count for each label per keyword
        for keyword in keywords:
            for label in labels:
                keyword_label_map[keyword.strip()][label.strip()] += 1
    
    return keyword_label_map

def compute_keyword_weights(keyword_label_map):
    keyword_weights = defaultdict(dict)
    
    # Track the number of labels each keyword appears in
    keyword_label_count = defaultdict(int)
    for keyword, label_counts in keyword_label_map.items():
        keyword_label_count[keyword] = len(label_counts)
    
    for keyword, label_counts in keyword_label_map.items():
        total_count = sum(label_counts.values())
        
        for label, count in label_counts.items():
            # Penalize keywords that appear in many labels by dividing their score by the number of labels they appear in
            keyword_weights[keyword][label] = (count / total_count) / keyword_label_count[keyword]
    
    return keyword_weights

def convert_to_label_keyword_view(keyword_weights):
    label_keyword_map = defaultdict(dict)
    
    # Invert the keyword-to-label mapping into a label-to-keyword mapping
    for keyword, labels in keyword_weights.items():
        for label, weight in labels.items():
            label_keyword_map[label][keyword] = weight
    
    return label_keyword_map

def display_label_keyword_mapping(label_keyword_map):
    print("\nLabel-to-Keyword Mapping with Inverse Scores:")
    for label, keywords in label_keyword_map.items():
        print(f"\nLabel: {label}")
        for keyword, score in keywords.items():
            print(f"  {keyword}: {score:.4f}")

            
def calculate_keyword_score_inverse(third_label, new_document_keywords, weight, label_keyword_map):
    score = 0
    keyword_importance = defaultdict(float)

    # Get the keywords associated with the third level label
    keywords = label_keyword_map.get(third_label, set())

    # Create a map of keywords to the number of third level labels they are linked to
    keyword_to_label_count = defaultdict(int)
    for label, keywords_in_label in label_keyword_map.items():
        for keyword in keywords_in_label:
            keyword_to_label_count[keyword] += 1

    for keyword in keywords:
        if keyword in new_document_keywords:
            # Get the total number of third-level labels this keyword is linked to
            label_count = keyword_to_label_count[keyword]
            
            # Adjust the weight based on the label count (reduce if the keyword appears in multiple third-level labels)
            if label_count > 0:
                adjusted_weight = weight / label_count
            else:
                adjusted_weight = 0
            
            # Accumulate the adjusted weight into the score
            score += adjusted_weight
            keyword_importance[keyword] = adjusted_weight

    return score, keyword_importance

def extract_keybert_keywords(text: str, top_n: int=10, ngram_range=(1, 3), stop_words=None, use_mmr=False, diversity=0.7):
    """
    Extract the top 'top_n' keywords from 'text' using a multilingual MiniLM model.

    :param text:        The input text (judicial document, etc.)
    :param top_n:       Number of keywords or phrases to return
    :param ngram_range: (min_n, max_n) for candidate keyword lengths
    :param use_mmr:     Whether to use Maximal Marginal Relevance for diversity
    :param diversity:   Diversity parameter for MMR (0 < diversity < 1)
    :return:            A list of (keyword, score) pairs
    """
    return kw_model.extract_keywords(
        text,
        top_n=top_n,
        keyphrase_ngram_range=ngram_range,   
        stop_words=stop_words,          
        use_mmr=use_mmr,                   
        diversity=diversity
    )

def seed_keybert_with_similar_docs(conn, new_doc_text, new_doc_embedding, artikel_id):
    """
    1. Retrieve top-N similar documents (already stored in DB) for 'new_doc_embedding'.
    2. Gather their keywords as 'seed_keywords'.
    3. Use KeyBERT with 'seed_keywords' to generate new keywords for the new doc.
    4. Update DB with the new doc's keywords.
    """
    top_similar = get_top_similar_items(conn, new_doc_embedding, top_n=5)

    seed_keywords = set()
    for row in top_similar:
        weighted_kw = row[2]  
        if weighted_kw:
            # If weighted_kw is a dict => keys are the keywords
            if isinstance(weighted_kw, dict):
                for kw in weighted_kw.keys():
                    seed_keywords.add(kw)
            else:
                # If it's a list of strings or something else, adapt accordingly
                # e.g.: for kw in weighted_kw: seed_keywords.add(kw)
                pass

    # Extract keywords from the new document with KeyBERT, seeding with the top-similar docs' keywords
    new_keywords_scored = kw_model.extract_keywords(
        new_doc_text,
        top_n=10,
        keyphrase_ngram_range=(1, 3),
        stop_words= ['en', 'van', 'de', 'een', 'het', 'dat', 'of', 'is', 'die', 'in', 'als', 'om'],
        use_mmr=True,
        diversity=0.8,
        seed_keywords=list(seed_keywords)
    )
    new_keywords_weighted = [{kw[0]: kw[1]} for kw in new_keywords_scored]

    return new_keywords_scored, new_keywords_weighted