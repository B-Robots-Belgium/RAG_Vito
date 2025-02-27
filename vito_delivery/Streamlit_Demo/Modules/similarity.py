# Modules/similarity.py

import numpy as np

from collections import defaultdict

def calculate_weighted_keyword_similarity(test_keywords, article_keywords, similarity=0.0):
    """
    Calculate the total weight of overlapping keywords between the test article's keywords and the database article's keywords.
    """
    # Convert keywords to lowercase for case-insensitive comparison
    test_keywords_lower = {kw.lower(): kw for kw in test_keywords.keys()}
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

    return similarity

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
            
            # Adjust the weight based on the label count
            if label_count > 0:
                adjusted_weight = weight / label_count
            else:
                adjusted_weight = 0
            
            # Accumulate the adjusted weight into the score
            score += adjusted_weight
            keyword_importance[keyword] = adjusted_weight

    return score, keyword_importance