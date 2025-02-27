# Modules/label_assignment.py

import pandas as pd
from collections import deque, defaultdict
from Modules.similarity import calculate_weighted_keyword_similarity
from Modules.utils import parse_pg_array, get_dynamic_weights
from Modules.database import get_top_similar_items, query_semantically_alike_items

def get_overlapping_keywords_info(stage, path, test_keywords, articles_df):
    """
    Collects information about overlapping keywords and other details for each article.

    Returns:
        list: A list of dictionaries containing the collected information.
    """
    info_list = []
    stage_header = f"Stage {stage}, Level: {' > '.join(path) if path else 'Initial items'}"
    for idx, row in articles_df.iterrows():
        article_info = {}
        article_info['stage'] = stage_header
        article_info['article_id'] = row['artikel']
        article_info['labels'] = row['labels']
        article_info['semantic_similarity'] = row['weight']
        article_info['combined_score'] = row['combined_score']
        article_info['test_keywords'] = list(test_keywords.keys())
        article_info['article_keywords'] = list(row['weighted_keywords'].keys())

        overlapping_keywords = []

        # Convert keywords to lowercase for case-insensitive comparison
        test_keywords_lower = {kw.lower(): kw for kw in test_keywords.keys()}
        article_keywords_lower = {kw.lower(): kw for kw in row['weighted_keywords'].keys()}

        # Check for substring overlaps between test and article keywords
        for test_kw_lower, test_kw_original in test_keywords_lower.items():
            for article_kw_lower, article_kw_original in article_keywords_lower.items():
                if test_kw_lower in article_kw_lower or article_kw_lower in test_kw_lower:
                    # Add the overlapping keywords and their weights
                    article_weight = row['weighted_keywords'][article_kw_original]
                    overlapping_keywords.append({
                        'test_keyword': test_kw_original,
                        'article_keyword': article_kw_original,
                        'article_weight': article_weight
                    })
                    # Stop after first match for this test_kw_lower
                    break  

        article_info['overlapping_keywords'] = overlapping_keywords

        info_list.append(article_info)

    return info_list

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
    print("---------------------------------------------------------------------------")
    print(f"\nStage {stage}, Level: {' > '.join(path) if path else 'Initial items'}")
    for idx, row in articles_df.iterrows():
        article_keywords = row['weighted_keywords']
        print(f"\nArticle ID: {row['artikel']}")
        print(f"Labels: {row['labels']}")
        print(f"Semantic similarity: {row['weight']}")
        print(f"Combined score: {row['combined_score']}\n")
        print("Test Article Keywords:", list(test_keywords.keys()))
        print("Article Keywords:", list(article_keywords.keys()))

        overlapping_keywords = []

        # Convert keywords to lowercase for case-insensitive comparison
        test_keywords_lower = {kw.lower(): kw for kw in test_keywords.keys()}
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
            print("Overlapping Keywords:")
            for overlap in overlapping_keywords:
                print(f"Test Keyword: '{overlap['test_keyword']}', "
                      f"Article Keyword: '{overlap['article_keyword']}', "
                      f"Article Keyword Weight: {overlap['article_weight']}")
        else:
            print("No overlapping keywords.")

def assign_labels_a_star(test_article, pg_vector_conn):
    """
    Assigns labels to the test_article using an A* search algorithm.
    Collects detailed information during the process.

    Returns:
        final_labels (list): The assigned labels.
        assigned_labels (list): Labels with their mean scores.
        outputs (list): Collected information for GUI display.
    """
    outputs = []

    ep = 1e-10

    # Get initial similar items
    initial_items = get_top_similar_items(pg_vector_conn, test_article.embedding, top_n=5)
    # Convert initial items to a DataFrame
    initial_df = pd.DataFrame(initial_items, columns=['artikel', 'chunk', 'weighted_keywords', 'labels', 'score'])
    initial_df['labels'] = initial_df['labels'].apply(parse_pg_array)
    initial_df['distance'] = 1 - initial_df['score']
    initial_df['weight'] = 1 / (initial_df['score'] + ep)

    # Use test article keywords
    test_weighted_keywords = test_article.weighted_keywords

    # Get dynamic weights depending on level
    alpha_initial, beta_initial = get_dynamic_weights(1)

    # Calculate initial keyword scores
    initial_df['keyword_score'] = initial_df['weighted_keywords'].apply(
        lambda article_keywords: calculate_weighted_keyword_similarity(
            test_weighted_keywords, article_keywords)
    )

    # Calculate initial combined scores
    initial_df['combined_score'] = alpha_initial * initial_df['weight'] + beta_initial * initial_df['keyword_score']

    # Collect outputs
    info = get_overlapping_keywords_info(stage=0, path=[], test_keywords=test_weighted_keywords, articles_df=initial_df)
    outputs.extend(info)

    # Begin the search
    assigned_labels = []
    max_depth = 3  # Maximum levels in hierarchy

    # Initialize the queue
    queue = deque()

    # Calculate initial label scores
    label_scores = defaultdict(list)
    label_combined_scores = defaultdict(float)
    label_dfs = defaultdict(pd.DataFrame)

    # Iterate over the initial items to collect scores and data per label
    for idx, row in initial_df.iterrows():
        if len(row['labels']) > 0:
            label = row['labels'][0]
            label_scores[label].append(row['combined_score'])
            label_combined_scores[label] = max(label_combined_scores.get(label, 0), row['combined_score'])
            label_dfs[label] = pd.concat([label_dfs[label], row.to_frame().T], ignore_index=True)

    # Calculate mean scores for labels
    label_mean_scores = {}
    for label in label_scores:
        mean_score = sum(label_scores[label]) / len(label_scores[label])
        label_mean_scores[label] = mean_score
        outputs.append({'type': 'label_mean_score', 'label': label, 'mean_score': mean_score})

    # Select only the label with the highest mean score at level 0 or take all top labels
    sorted_labels = sorted(label_mean_scores.items(), key=lambda x: x[1], reverse=True)
    top_mean_score = sorted_labels[0][1]
    threshold = top_mean_score * 0.90
    top_labels = [label for label, score in sorted_labels if score >= threshold]
    for idx, label in enumerate(top_labels):
        top_label = sorted_labels[0][0]
        mean_score = label_mean_scores[top_label]
        df_filtered = label_dfs[top_label]

        # Add initial paths to the queue
        queue.append({
            'path': [top_label],
            'mean_score': mean_score,
            'level': 1,
            'df': df_filtered
        })

    # Perform the search
    while queue:
        current = queue.popleft()
        path = current['path']
        level = current['level']
        df = current['df']
        level_mean_score = current['mean_score']

        # Collect outputs
        info = get_overlapping_keywords_info(stage=level, path=path, test_keywords=test_weighted_keywords, articles_df=df)
        outputs.extend(info)

        # Prepare label index for current level
        label_index = level

        alpha, beta = get_dynamic_weights(level)

        # Calculate scores for the current level
        label_scores = defaultdict(list)
        label_combined_scores = defaultdict(float)
        label_dfs = defaultdict(pd.DataFrame)

        # Collect label scores and DataFrames at the current level
        for idx, row in df.iterrows():
            if len(row['labels']) > label_index:
                label = row['labels'][label_index]
                combined_score = row['combined_score']
                label_scores[label].append(combined_score)
                label_combined_scores[label] = max(label_combined_scores.get(label, 0), combined_score)
                label_dfs[label] = pd.concat([label_dfs[label], row.to_frame().T], ignore_index=True)
        
        outputs.append({'type': 'label_scores', 'level': level, 'label_scores': label_scores})
        if label_scores:
            # Calculate mean scores for labels at current level
            label_mean_scores = {}
            for label in label_scores:
                mean_score = sum(label_scores[label]) / len(label_scores[label])
                label_mean_scores[label] = mean_score
                outputs.append({'type': 'label_mean_score', 'label': label, 'mean_score': mean_score})
        else:
            outputs.append({'type': 'info', 'message': "No further labels at this level."})
            assigned_labels.append({
                'path': path,
                'mean_score': level_mean_score
            })
            continue
            
        if level >= max_depth:
            assigned_labels.append({
                'path': path,
                'mean_score': level_mean_score
            })
            continue

        # For levels beyond the first, include labels within a % of the top score
        # if level == 1:
        #     # At first level, only select the label with the highest mean score
        #     sorted_labels = sorted(label_mean_scores.items(), key=lambda x: x[1], reverse=True)
        #     top_label = sorted_labels[0][0]
        #     top_mean_score = sorted_labels[0][1]
        #     top_labels = [top_label]
        # else:
            # For other levels, include labels within a % threshold
        sorted_labels = sorted(label_mean_scores.items(), key=lambda x: x[1], reverse=True)
        top_mean_score = sorted_labels[0][1]
        threshold = top_mean_score * 0.80
        top_labels = [label for label, score in sorted_labels if score >= threshold]

        for label in top_labels:
            new_path = path + [label]
            new_mean_score = label_mean_scores[label]
            new_df = label_dfs[label]

            # Re-query to get new DataFrame for the next level
            current_label_path = "\\".join(new_path)
            outputs.append({'type': 'requery', 'label_path': current_label_path})
            
            new_items = query_semantically_alike_items(pg_vector_conn, test_article.embedding, current_label_path, top_n=5*level)

            if not new_items or level + 1 >= max_depth:
                assigned_labels.append({
                    'path': new_path,
                    'mean_score': new_mean_score
                })
                continue

            # Convert new items to a DataFrame
            new_df = pd.DataFrame(new_items, columns=['artikel', 'chunk', 'weighted_keywords', 'labels', 'score'])
            new_df['labels'] = new_df['labels'].apply(parse_pg_array)
            new_df['distance'] = 1 - new_df['score']
            new_df['weight'] = 1 / (new_df['score'] + ep)

            # Calculate keyword scores again for new query results
            new_df['keyword_score'] = new_df['weighted_keywords'].apply(
                lambda article_keywords: calculate_weighted_keyword_similarity(
                    test_weighted_keywords, article_keywords
                )
            )

            # Calculate combined scores for new items
            new_df['combined_score'] = alpha * new_df['weight'] + beta * new_df['keyword_score']

            queue.append({
                'path': new_path,
                'mean_score': new_mean_score,
                'level': level + 1,
                'df': new_df
            })

    # After the search select the paths with the highest scores
    assigned_labels = sorted(assigned_labels, key=lambda x: x['mean_score'], reverse=True)

     # Determine the threshold for including multiple labels
    if assigned_labels:
        top_mean_score = assigned_labels[0]['mean_score']
        threshold = top_mean_score * 0.90 

        # Collect all paths whose mean score is within 10% of the top score
        outputs.append({'type': 'assigned_label', 'top_mean_score': top_mean_score})
        final_labels = [item['path'] for item in assigned_labels if item['mean_score'] >= threshold]
    else:
        final_labels = []

    # Collect assigned labels with mean scores
    outputs.append({'type': 'final_labels', 'assigned_labels': assigned_labels})

    return final_labels, assigned_labels, outputs