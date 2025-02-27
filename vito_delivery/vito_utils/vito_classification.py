from collections import defaultdict, deque
from .db_actions import (
    get_top_similar_items, 
    query_sematically_alike_items)
from .utils import (
    parse_pg_array, 
    get_dynamic_weights)
from .keyword_utils import (
    calculate_weighted_keyword_similarity, 
    print_overlapping_keywords)
from .vito_classes import VitoArticle, VitoBoek
import pandas as pd
import psycopg2

from .api_calls import retrieve_artikel, retrieve_boek_from_artikel

def move_label_up(assigned_labels, highest_label, current_label_path, hierarchical_labels):
    assigned_labels.append(highest_label.split('\\')[-1])
    # Update the hierarchical label path for the next level
    if current_label_path:
        current_label_path = current_label_path + "\\" + highest_label
    else:
        current_label_path = highest_label
    # Get the next level of allowed labels 
    allowed_sub_labels = hierarchical_labels.get(current_label_path, [])
    if allowed_sub_labels:
        allowed_labels = allowed_sub_labels
    else:
        # If there are no sub-labels stop going deeper but continue from other top-level labels
        allowed_labels = []
    return current_label_path, allowed_labels

def assign_labels_a_star(testArticle: VitoArticle, PgVectorConn, max_depth: int = 3) -> list:
    """
    Assigns labels to the test_article using an A* search algorithm.
    Collects detailed information during the process.

    Returns:
        final_labels (list): The assigned labels.
        assigned_labels (list): Labels with their mean scores.
        outputs (list): Collected information for GUI display.
    """
    # Small constant to avoid division by zero
    ep = 1e-10  

    # Start by querying the top-level items
    initial_items = get_top_similar_items(PgVectorConn, testArticle.embedding, 5)
    # Convert initial items to a DataFrame
    initial_df = pd.DataFrame(initial_items, columns=['artikel', 'chunk', 'weighted_keywords', 'labels', 'score'])
    initial_df['labels'] = initial_df['labels'].apply(parse_pg_array)
    initial_df['distance'] = 1 - initial_df['score']
    initial_df['weight'] = 1 / (initial_df['score'] + ep)

    # Use new document keywords
    test_weighted_keywords = testArticle.weighted_keywords

    # Get dynamic weights depending on level
    alpha_initial, beta_initial = get_dynamic_weights(1)

    # Calculate initial keyword scores
    initial_df['keyword_score'] = initial_df['weighted_keywords'].apply(
    lambda article_keywords: calculate_weighted_keyword_similarity(
        test_weighted_keywords, article_keywords)
    )

    initial_df['extra_context_score'] = 0.0

    for idx, row in initial_df.iterrows():
        article_id = str(row['artikel'])

        try:
            artikel_data = retrieve_artikel(article_id)

            boek_id = artikel_data['metadata'][0]['href'].split('/')[-1]

            linked_boek = VitoBoek(retrieve_boek_from_artikel(boek_id))

            linked_boek.get_boek_info()

            print("Boek samenvatting: " + linked_boek.samenvatting + "\n")
            print("Boek titel: " + linked_boek.titel + "\n")

        except Exception as e:
            print(f"Error retrieving or processing data for artikel {article_id}: {e}")
            continue

    # Calculate initial combined scores
    initial_df['combined_score'] = alpha_initial * initial_df['weight'] + beta_initial * initial_df['keyword_score']

    print_overlapping_keywords(stage=0, path=[], test_keywords=test_weighted_keywords, articles_df=initial_df)

    # Begin the search
    assigned_labels = []
    # Maximum levels in hierarchy
    max_depth = max_depth  

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
        print(f"\nLabel: {label}, Mean Score: {mean_score}")

    # Select only the label with the highest mean score at level 0 or take all top labels
    sorted_labels = sorted(label_mean_scores.items(), key=lambda x: x[1], reverse=True)
    for idx, label in enumerate(sorted_labels):
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

        print_overlapping_keywords(stage=level, path=path, test_keywords=test_weighted_keywords, articles_df=df)

        # Prepare label index for current level
        label_index = level

        alpha, beta = get_dynamic_weights(level)

        # Initialize scores for the current level
        label_scores = defaultdict(list)
        label_combined_scores = defaultdict(float)
        label_dfs = defaultdict(pd.DataFrame)

        # Collect label scores and Dataframes at the current level
        for idx, row in df.iterrows():
            if len(row['labels']) > label_index:
                label = row['labels'][label_index]
                combined_score = row['combined_score']
                label_scores[label].append(combined_score)
                label_combined_scores[label] = max(label_combined_scores.get(label, 0), combined_score)
                label_dfs[label] = pd.concat([label_dfs[label], row.to_frame().T], ignore_index=True)
        
        print(f"\nLabel scores for level {level}: {label_scores}")
        if label_scores:
            # Calculate mean scores for labels at current level
            label_mean_scores = {}
            for label in label_scores:
                mean_score = sum(label_scores[label]) / len(label_scores[label])
                label_mean_scores[label] = mean_score
                print(f"Label: {label}, mean Score: {mean_score:.4f}")
        else:
            print("No further labels at this level.")
            # Since there are no further labels, we can append the current path
            assigned_labels.append({
                'path': path,
                'mean_score': level_mean_score
            })
            continue
            
        if level >= max_depth:
            # Reached maximum depth or no further labels, add to assigned labels
            assigned_labels.append({
                'path': path,
                'mean_score': level_mean_score
            })
            continue

        # For levels beyond the first, include labels within a % of the top score
        if level == 1:
            # At first level, only select the label with the highest mean score
            sorted_labels = sorted(label_mean_scores.items(), key=lambda x: x[1], reverse=True)
            top_label = sorted_labels[0][0]
            top_mean_score = sorted_labels[0][1]
            top_labels = [top_label]
        else:
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
            print(f"Re-querying with label path: {current_label_path}")
            
            new_items = query_sematically_alike_items(PgVectorConn, testArticle.embedding, current_label_path, top_n=5*level)

            if not new_items or level + 1 >= max_depth:
                # No further items, assign labels
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


    # After the search, select the paths with the highest scores
    assigned_labels = sorted(assigned_labels, key=lambda x: x['mean_score'], reverse=True)

     # Determine the threshold for including multiple labels
    if assigned_labels:
        top_mean_score = assigned_labels[0]['mean_score']
        threshold = top_mean_score * 0.90

        # Collect all paths whose mean score is within 5% of the top score
        print("Assigned Label for top mean score: " + str(top_mean_score))
        final_labels = [item['path'] for item in assigned_labels if item['mean_score'] >= threshold]
    else:
        final_labels = []

    # Print the assigned labels with their mean scores
    print("\nAssigned Labels with mean scores:")
    for item in assigned_labels:
        print(f"Labels: {item['path']}, Mean Score: {item['mean_score']:.4f}")

    return final_labels, assigned_labels