import os
import json
import random
from vito_utils.vito_classes import VitoArticle

def load_text_from_folder(folder_path):
    """Load all JSON files from the given folder and return a list of articles (artikel attribute)."""
    articles = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'artikel' in data:
                            articles.append(data['artikel'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {file}: {e}")
    return articles

def load_articles_from_folder(folder_path):
    """Load all JSON files from the given folder and return a list of articles (artikel attribute)."""
    articles = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                try:
                    data = VitoArticle.load_from_json(os.path.join(root, file))
                    articles.append(data)  
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {file}: {e}")
    return articles

def get_unique_articles_from_folders(folders, comparison_folder):
    """Get articles from multiple folders and filter out the ones that exist in the comparison folder."""
    
    # Load all articles from the comparison folder
    comparison_articles = set(load_text_from_folder(comparison_folder))

    # Load articles from the target folders
    unique_articles = []
    for folder in folders:
        folder_articles = load_articles_from_folder(folder)
        
        # Check if the article exists in the comparison folder
        for artikel in folder_articles:
            if artikel.artikel not in comparison_articles:
                unique_articles.append(artikel)

    return unique_articles

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
                test_article = VitoArticle(os.path.join(root, file))
                List_Articles.append(test_article)
    return List_Articles

def get_new_articles(data_folders, article_folder):
    # Get all articles from the provided folders
    List_Articles = get_articles_from_folders(data_folders)

    # Load existing articles from Article_Objects folder
    existing_articles = load_existing_articles(article_folder)

    # Filter new articles that are not in the existing database and where 'inhoud' isn't empty after cleaning
    new_articles = []
    for article in List_Articles:
        if not check_article_existence(article, existing_articles):
            article_json = article
            if article_json.inhoud:  # Only add articles with non-empty content
                new_articles.append(article_json)

    # Return new articles in the required JSON format
    return new_articles

def get_hierarchical_labels_from_folder(root_folder):
    hierarchical_labels = {}

    for dirpath, dirnames, filenames in os.walk(root_folder):
        relative_path = os.path.relpath(dirpath, root_folder)
        if relative_path == '.':
            relative_path = os.path.basename(root_folder)
        hierarchical_labels[relative_path] = dirnames

    return hierarchical_labels

def is_leaf_directory(dir_path):
    """Return True if 'dir_path' has no subdirectories."""
    for entry in os.scandir(dir_path):
        if entry.is_dir():
            return False
    return True


def get_leaf_directories(base_folder):
    """
    Recursively walk through 'base_folder' to find all the leaf directories (i.e., 
    directories that have no subdirectories).
    """
    leaf_dirs = []
    for root, dirs, _ in os.walk(base_folder):
        if not dirs:
            leaf_dirs.append(root)
    return leaf_dirs


def get_unique_articles_from_folders_with_split(folders, comparison_folder, test_ratio=0.2, seed=42) -> list[VitoArticle]:
    """
    1. Find all leaf folders in 'folders'.
    2. Load all articles from each leaf folder, filtering out duplicates found in 'comparison_folder'.
    3. Shuffle and split each leaf folder’s articles into train/test at 80:20 (by default).
    
    :param folders: List of folder paths to scan for articles.
    :param comparison_folder: Folder path whose articles are considered "already known".
    :param test_ratio: The fraction of leaf-folder articles to reserve for testing (default 0.2).
    :param seed: Random seed for reproducible shuffling.
    :return: (train_articles, test_articles) as lists of VitoArticle objects.
    """

    # Load articles from the comparison folder as a set of artikel strings
    comparison_articles = set(load_text_from_folder(comparison_folder))

    # Prepare final containers
    train_articles = []
    test_articles = []

    random.seed(seed)  # for reproducible splits

    # For each folder that we want to process
    for folder in folders:
        # Find all leaf directories in this folder
        leaf_dirs = get_leaf_directories(folder)

        for leaf_dir in leaf_dirs:
            # Load all VitoArticles from this leaf directory
            leaf_articles = load_articles_from_folder(leaf_dir)

            # Filter out articles that exist in the comparison folder (by 'artikel' text)
            new_leaf_articles = [
                a for a in leaf_articles 
                if a.artikel not in comparison_articles
            ]

            # Shuffle them (so splitting is random)
            random.shuffle(new_leaf_articles)

            # Compute split index
            split_index = int(len(new_leaf_articles) * (1 - test_ratio))

            # 80% => DB, 20% => test (by default)
            leaf_train = new_leaf_articles[:split_index]
            leaf_test = new_leaf_articles[split_index:]

            # Add them to the global lists
            train_articles.extend(leaf_train)
            test_articles.extend(leaf_test)

    return train_articles, test_articles