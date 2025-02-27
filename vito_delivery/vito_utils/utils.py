from bs4 import BeautifulSoup
import re
from io import StringIO
import csv
import os
from .keyword_utils import extract_keybert_keywords
import json

def clean_html(raw_html):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()
    
    # Remove other unwanted characters using regex
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'&#160;|&nbsp;', ' ', text)  # Replace non-breaking spaces with regular space
    text = re.sub(r'[^\w\s.,-]', '', text)  # Remove special characters, keeping punctuation
    
    return text.strip()

def parse_pg_array(array_str):
    trimmed = array_str.strip('{}')
    f = StringIO(trimmed)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        return [item.strip() for item in row]

def parse_labels(labels_str):
    # Remove the curly braces and split by comma
    labels_str = labels_str.strip('{}')
    labels_list = labels_str.split(',')
    # Strip whitespace
    labels_list = [label.strip() for label in labels_list]
    return labels_list

def get_dynamic_weights(level):
        if level == 1:
            return 1.0, 0.5
        elif level == 2:
            return 1.0, 1.0
        elif level == 3:
            return 0.5, 2.0
        elif level == 4:
            return 0.5, 3.0
        elif level == 5:
            return 0.5, 4.0

def update_db_keywords_and_title(json_folder, conn):
    """
    For every .json file in `json_folder`, parse the artikel_id and artikel (title),
    generate new keywords via KeyBERT, and then update the relevant rows in your
    PostgreSQL table for that artikel_id.
    """
    count = 0
    # 2) Traverse the directory
    for filename in os.listdir(json_folder)[800:900]:
        if not filename.endswith(".json"):
            continue
        
        json_path = os.path.join(json_folder, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping {filename}; JSON decode error: {e}")
                continue
            
            # 3) Extract artikel_id and artikel (title)
            artikel_id = data.get("artikel id")
            artikel_title = data.get("artikel")
            inhoud = data.get("inhoud", "")

            # If necessary, skip empty or invalid items
            if not artikel_id or not artikel_title:
                print(f"Skipping {filename}; missing artikel_id or artikel.")
                continue

            # 4) Clean up the text, then generate new keywords
            cleaned_text = clean_html(inhoud)

            unwanted_html = ["<div>\n     <div>[...]</div>\n   </div>", "<div>\r\n     <div>[...]</div>\r\n   </div>", "<div>\r\n     <div>[...]</div>]</div>", "<p>[...]</p>"]

            to_delete = False

            if inhoud.strip() in unwanted_html:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            DELETE FROM vito_article
                                  WHERE artikel_id = %s
                            """,
                            (str(artikel_id),)  # cast to string if artikel_id is text in DB
                        )
                    conn.commit()
                    to_delete = True
                    print(f"Deleted artikel_id={artikel_id} (file: {filename}) because inhoud matched unwanted HTML.")
                except Exception as e:
                    conn.rollback()
                    print(f"Error deleting artikel_id={artikel_id}: {e}")
            else:
                print(f"artikel_id={artikel_id} (file: {filename}) does NOT match unwanted_html; not deleted.")

            # KeyBERT returns list of (keyword, score)
            new_keywords_scored = extract_keybert_keywords(cleaned_text)
            # We only want the keyword strings
            new_keywords_list = [kw for kw, score in new_keywords_scored]

            # 5) Update the DB
            if to_delete == False:
                try:
                    with conn.cursor() as cur:
                        # If an artikel can have multiple chunk_id rows, we update them all
                        # in one statement. Adjust your_table_name as needed.
                        cur.execute(
                            """
                            UPDATE vito_article
                            SET keywords       = %s,
                                artikel_title  = %s
                            WHERE artikel_id     = %s
                            """,
                            (new_keywords_list, artikel_title, str(artikel_id))
                        )
                    conn.commit()
                    print(f"Updated artikel_id={artikel_id} with {len(new_keywords_list)} keywords.")
                except Exception as e:
                    conn.rollback()
                    print(f"Error updating artikel_id={artikel_id}: {e}")
        count= count + 1
        print(count)