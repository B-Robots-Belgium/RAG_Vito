from vito_utils.utils import clean_html
from vito_utils.vito_classes import VitoArticle
from vito_utils.vito_classification import assign_labels_a_star
from vito_utils.keyword_utils import extract_keybert_keywords, seed_keybert_with_similar_docs
from vito_utils.vito_llm import extract_with_langchain
from vito_utils.loading import get_unique_articles_from_folders_with_split, load_articles_from_folder
import openai
import os
from pathlib import Path
import json
import random
import time
import statistics
import psycopg2
import logging
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'logs/general/{datetime.datetime.now().strftime("%Y%m%d")}_updates.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

def calculate_label_accuracy_per_level(data, n_levels: int = 3):
    # Initialize counters for correct predictions at each label level
    top1_label_level_counts = {}  
    topn_label_level_counts = {}  
    incorrect_items = []
    detailed_results = []

    # Initialize lists for score collection
    top1_scores = []
    topn_scores = []

    # Iterate through each item
    for item in data:
        result_labels_list = item['List_result_labels'] 
        correct_labels = item['correct_labels']          
        score_chosen_items = item.get('score_chosen_items', 0.0)  
        max_levels = n_levels

        # Initialize the level in label_level_counts if not already
        for level in range(max_levels):
            if level not in top1_label_level_counts:
                top1_label_level_counts[level] = {'correct': 0, 'total': 0}
            if level not in topn_label_level_counts:
                topn_label_level_counts[level] = {'correct': 0, 'total': 0}

        # Top-1 Accuracy
        # Use the first classification result
        if result_labels_list:
            top1_result_labels = result_labels_list[0] 
            top1_correct = True 

            # Iterate through each level and compare the labels
            for level in range(max_levels):
                result_label = top1_result_labels[level] if level < len(top1_result_labels) else None
                correct_label = correct_labels[level] if level < len(correct_labels) else None

                # Compare the labels at the current level
                if result_label == correct_label:
                    top1_label_level_counts[level]['correct'] += 1
                else:
                    top1_correct = False

                # Count total comparisons made at this level
                top1_label_level_counts[level]['total'] += 1

            # If all labels match up to max_levels, consider the item correct
            if top1_correct:
                top1_scores.append(score_chosen_items)
                item['top1_correct'] = True
                if len(result_labels_list) == 1:
                    print("only one item returned")
                    item['stp'] = True
            else:
                item['top1_correct'] = False
                if len(result_labels_list) == 1:
                    print("only one item returned, is wrong")
                    incorrect_items.append({
                    "Artikel_id": item["Artikel_id"],
                    "Artikel_name": item["Artikel_name"],
                    "Artikel_keywords": item["Artikel_keywords"]
                    })
                    item['stp'] = False
        else:
            # No result labels returned
            for level in range(max_levels):
                top1_label_level_counts[level]['total'] += 1  
            item['top1_correct'] = False
            if len(result_labels_list) == 1:
                    print("only one item returned, is wrong")
                    item['stp'] = False

        # Top-N Accuracy
        # Check if any of the classifications have the correct label at each level
        for level in range(max_levels):
            correct_label = correct_labels[level] if level < len(correct_labels) else None
            match_found_at_level = False

            for result_labels in result_labels_list:
                result_label = result_labels[level] if level < len(result_labels) else None
                if result_label == correct_label:
                    match_found_at_level = True
                    break

            if match_found_at_level:
                topn_label_level_counts[level]['correct'] += 1

            # Count total comparisons made at this level
            topn_label_level_counts[level]['total'] += 1

        # Check for overall correctness in Top-N
        # We consider an item correct if any of the result_labels exactly match correct_labels up to max_levels
        topn_correct = False
        if result_labels_list:
            for result_labels in result_labels_list:
                match_found = True
                for level in range(max_levels):
                    result_label = result_labels[level] if level < len(result_labels) else None
                    correct_label = correct_labels[level] if level < len(correct_labels) else None
                    if result_label != correct_label:
                        match_found = False
                        break
                if match_found:
                    topn_scores.append(score_chosen_items)
                    topn_correct = True
                    item['topn_correct'] = True
                    break
            if not topn_correct:
                incorrect_items.append({
                    "Artikel_id": item["Artikel_id"],
                    "Artikel_name": item["Artikel_name"],
                    "Artikel_keywords": item["Artikel_keywords"]
                })
                item['topn_correct'] = False
        else:
            incorrect_items.append({
                    "Artikel_id": item["Artikel_id"],
                    "Artikel_name": item["Artikel_name"],
                    "Artikel_keywords": item["Artikel_keywords"]
                })
            item['topn_correct'] = False

        # Calculate individual confidence scores
        # Confidence is the score_chosen_items if correct, else 0
        item['confidence_top1'] = score_chosen_items if item['top1_correct'] else 0.0
        item['confidence_topn'] = score_chosen_items if item['topn_correct'] else 0.0

        

        if len(result_labels_list) > 1 or len(result_labels_list) < 1:
            detailed_results.append({
            'Artikel_id': item['Artikel_id'],
            'Artikel_name': item['Artikel_name'],
            'result_labels': result_labels_list,
            'correct_labels': correct_labels,
            'score_chosen_items': score_chosen_items,
            'top1_correct': item['top1_correct'],
            'topn_correct': item['topn_correct'],
            'confidence_top1': item['confidence_top1'],
            'confidence_topn': item['confidence_topn'],
            'straight_through_rate': False
            })
        else:
            detailed_results.append({
            'Artikel_id': item['Artikel_id'],
            'Artikel_name': item['Artikel_name'],
            'result_labels': result_labels_list,
            'correct_labels': correct_labels,
            'score_chosen_items': score_chosen_items,
            'top1_correct': item['top1_correct'],
            'topn_correct': item['topn_correct'],
            'confidence_top1': item['confidence_top1'],
            'confidence_topn': item['confidence_topn'],
            'straight_through_rate': item['stp']
            })

    # Calculate Top-1 accuracy per label level
    top1_accuracy_per_label_level = {
        level: (count['correct'] / count['total']) * 100 if count['total'] > 0 else 0
        for level, count in top1_label_level_counts.items()
    }

    # Calculate Top-N accuracy per label level
    topn_accuracy_per_label_level = {
        level: (count['correct'] / count['total']) * 100 if count['total'] > 0 else 0
        for level, count in topn_label_level_counts.items()
    }

    count_stp_true = sum(item['straight_through_rate'] for item in detailed_results)
    percentage_stp_true = 100 * count_stp_true / len(detailed_results) if detailed_results else 0

    # Calculate median scores
    median_top1_score = statistics.median(top1_scores) if top1_scores else 0.0
    median_topn_score = statistics.median(topn_scores) if topn_scores else 0.0

    if len(incorrect_items) > 0:
        with open(f'{Path(__file__).parent.absolute()}\\logs\\incorrect_items\\{datetime.datetime.now().strftime("%Y%m%d")}.json', 'w') as fp:
            json.dump(incorrect_items, fp=fp)

    return detailed_results, top1_accuracy_per_label_level, topn_accuracy_per_label_level, median_top1_score, median_topn_score, percentage_stp_true, incorrect_items

folders_to_check = ["data/Stoffen", "data/Handhaving", "data/Compartiment"]
comparison_folder = "Article_Objects"

unwanted_html = ["<div>\n     <div>[...]</div>\n   </div>", 
                 "<div>\r\n     <div>[...]</div>\r\n   </div>", 
                 "<div>\r\n     <div>[...]</div>]</div>", 
                 "<p>[...]</p>"]

if __name__ == "__main__": 
    openaiClient = openai.OpenAI(
    api_key= os.getenv("OPENAI_API_KEY"),
    organization= os.getenv("OPENAPI_ORG")
    )

    PgVectorConn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )

    list_result_labels = []

    # Run code here
    # ----------------------------------------------------------

    test_db = load_articles_from_folder(f'{Path(__file__).parent.absolute()}\\data\\test_db')
    random.shuffle(test_db)

    for article in test_db[:10]:
        if article.inhoud in unwanted_html:
            print(30 * '-')
            logging.error("Skipping due to invalid HTML/article")
        else:
            try:
                article.inhoud = clean_html(article.inhoud)
                article.embedding = article.get_embedding(openaiClient)
                article.keywords, article.weighted_keywords = seed_keybert_with_similar_docs(PgVectorConn, article.inhoud, article.embedding, article.artikel_id)
                LangChain_response, cb_openai = extract_with_langchain(article=article)
                print(article.artikel_id)
                print(cb_openai)
                print(LangChain_response)
                if LangChain_response.is_addendum == False:
                    result_labels, assigned_labels = assign_labels_a_star(testArticle=article, 
                                                                          PgVectorConn=PgVectorConn,
                                                                          initial_item_amount=10,
                                                                          semantic_item_amount=7,
                                                                          include_label_threshold=0.9,
                                                                          multiple_label_threshold=0.95)
                    list_result_labels.append({
                        "Artikel_id": article.artikel_id, 
                        "Artikel_name": article.artikel, 
                        "List_result_labels": result_labels, 
                        "correct_labels": article.labels,
                        "Artikel_keywords": article.weighted_keywords})
                else:
                    try:
                        logging.info(LangChain_response.referenties[0])
                    except Exception as e:
                        logging.error("No reference to another article was found")
                print(30 * '-')
                logging.info(f"Article: {article.artikel_id}, Labels: {result_labels}, correct labels: {article.labels}")
            except Exception as e:
                logging.error(f"Article: {article.artikel_id}, {e}")
        
    # Calculate accuracies and median scores
    detailed_results, top1_accuracy, topn_accuracy, median_top1_score, median_topn_score, percentage_stp_rate, list_incorrect_items = calculate_label_accuracy_per_level(list_result_labels)

    # Print Top-1 Accuracy per Label Level
    print("Top-1 Accuracy per Label Level:")
    for level, accuracy in top1_accuracy.items():
        print(f"Level {level}: {accuracy:.2f}%")
    print(f"Median Top-1 Score: {median_top1_score:.4f}")

    # Print Top-N Accuracy per Label Level
    print("\nTop-N Accuracy per Label Level:")
    for level, accuracy in topn_accuracy.items():
        print(f"Level {level}: {accuracy:.2f}%")
    print(f"Median Top-N Score: {median_topn_score:.4f}")

    print(f"Straight Throughput Rate: {percentage_stp_rate}")

    # --------------------------------------------------------
    # End of executable code

    PgVectorConn.close()

        # to_db = load_articles_from_folder(f'{Path(__file__).parent.absolute()}\\to_db')
    # testing_items = to_db[250:]
    # for testing_item in testing_items:
    #     testing_item.generate_embeddings_with_error_handling(openaiClient)
    #     testing_item.extract_keywords()
    #     testing_item.store_embedding_in_pgvector(PgVectorConn)

    

# JSON_FOLDER = 'C:/Users/EhranLenaerts/Documents/RAG_B_Robots/RAG_Vito/vito_delivery/Article_Objects'

# update_db_keywords_and_title(JSON_FOLDER, PgVectorConn)

# def update_db_keywords_and_title(json_folder, conn):
#     """
#     For every .json file in `json_folder`, parse the artikel_id and artikel (title),
#     generate new keywords via KeyBERT, and then update the relevant rows in your
#     PostgreSQL table for that artikel_id.
#     """

#     # 2) Traverse the directory
#     for filename in os.listdir(json_folder)[650:]:
#         if not filename.endswith(".json"):
#             continue
        
#         json_path = os.path.join(json_folder, filename)
#         with open(json_path, "r", encoding="utf-8") as f:
#             try:
#                 data = json.load(f)
#             except json.JSONDecodeError as e:
#                 print(f"Skipping {filename}; JSON decode error: {e}")
#                 continue
            
#             # 3) Extract artikel_id and artikel (title)
#             artikel_id = data.get("artikel id")
#             artikel_title = data.get("artikel")
#             inhoud = data.get("inhoud", "")

#             # If necessary, skip empty or invalid items
#             if not artikel_id or not artikel_title:
#                 print(f"Skipping {filename}; missing artikel_id or artikel.")
#                 continue

#             # 4) Clean up the text, then generate new keywords
#             cleaned_text = clean_html(inhoud)

#             # KeyBERT returns list of (keyword, score) pairs
#             new_keywords_scored = extract_keybert_keywords(cleaned_text)

#             # Normal keywords list
#             new_keywords_list = [kw for kw, score in new_keywords_scored]

#             # Build a dictionary of {keyword: score}
#             weighted_keywords_dict = {kw: float(score) for kw, score in new_keywords_scored}

#             # 5) Update the DB
#             try:
#                 with conn.cursor() as cur:
#                     # Note we also update weighted_keywords
#                     cur.execute(
#                         """
#                         UPDATE vito_article
#                         SET keywords          = %s,
#                             weighted_keywords = %s,
#                             artikel_title     = %s
#                         WHERE artikel_id        = %s
#                         """,
#                         (new_keywords_list, json.dumps(weighted_keywords_dict), artikel_title, str(artikel_id))
#                     )
#                 conn.commit()
#                 print(f"Updated artikel_id={artikel_id} with {len(new_keywords_list)} keywords.")
#             except Exception as e:
#                 conn.rollback()
#                 print(f"Error updating artikel_id={artikel_id}: {e}")

    # train, test = get_unique_articles_from_folders_with_split(folders_to_check, comparison_folder)

    # print(len(train))
    # print(len(test))

    # for item in train:
    #     item.save_to_json(f"{Path(__file__).parent.absolute()}\\to_db\\{item.artikel_id}.json")

    # for item in test:
    #     item.save_to_json(f"{Path(__file__).parent.absolute()}\\test_db\\{item.artikel_id}.json")