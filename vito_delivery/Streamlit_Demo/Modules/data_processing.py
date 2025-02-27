# Modules/data_processing.py

import re
import os
import json
from bs4 import BeautifulSoup
from Modules.article import VitoArticle
import pandas as pd

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

def load_existing_articles(article_folder):
    existing_articles = []
    for root, _, files in os.walk(article_folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as json_file:
                    try:
                        article_data = json.load(json_file)
                        existing_articles.append(article_data)
                    except json.JSONDecodeError as e:
                        print(f"Error loading JSON from file {file}: {e}")
    return existing_articles

def check_article_existence(article, existing_articles):
    for existing_article in existing_articles:
        if article.artikel_id == existing_article.get("artikel id") or article.artikel == existing_article.get("artikel"):
            return True
    return False

def get_articles_from_folders(folder_paths):
    List_Articles = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    test_article = VitoArticle(os.path.join(root, file))
                    List_Articles.append(test_article)
    return List_Articles

def get_new_articles(data_folders, article_folder):
    # Step 1: Get all articles from the provided folders
    List_Articles = get_articles_from_folders(data_folders)

    # Step 2: Load existing articles from article_folder
    existing_articles = load_existing_articles(article_folder)

    # Step 3: Filter new articles that are not in the existing database and whose 'inhoud' isn't empty after cleaning
    new_articles = []
    for article in List_Articles:
        if not check_article_existence(article, existing_articles):
            if article.inhoud:  # Only add articles with non-empty content
                new_articles.append(article)

    return new_articles

def get_hierarchical_labels_from_folder(root_folder):
    hierarchical_labels = {}

    for dirpath, dirnames, filenames in os.walk(root_folder):
        relative_path = os.path.relpath(dirpath, root_folder)
        if relative_path == '.':
            relative_path = os.path.basename(root_folder)
        hierarchical_labels[relative_path] = dirnames

    return hierarchical_labels