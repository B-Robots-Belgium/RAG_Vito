import requests
import time
import json
import os
from .utils import clean_html
from .vito_classes import VitoBoek

urls = []

dirname = os.path.dirname(__file__)

with open(f"{dirname}/local_metadata/urls.json", "r") as f:
    urls = json.load(f)

def retrieve_entire_structure():
    """
    Retrieves the entire structure from the main structuur navigator using the specified URL.
    
    This function makes an authenticated GET request to the URL obtained from the 
    'main_structuur_navigator' entry in the 'urls' dictionary. The credentials for 
    authentication are retrieved from environment variables.

    Returns:
        The response from the GET request.
    """

    url = urls["main_structuur_navigator"]["url"]
    return requests.get(url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()

def retrieve_artikel(artikel_id: str):
    """
    Retrieves an artikel metadata from the artikel navigator using the specified artikel ID.
    
    This function makes an authenticated GET request to the URL obtained from the 
    'artikel_navigator' entry in the 'urls' dictionary, appending the artikel ID 
    to the end of the URL. The credentials for authentication are retrieved from 
    environment variables.

    Args:
        artikel_id (str): The ID of the artikel to retrieve.

    Returns:
        The response from the GET request.
    """

    url = urls["artikel_navigator"]["url"] + artikel_id
    return requests.get(url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()

def retrieve_artikel_versies(artikel_id: str):
    """
    Retrieves an artikel version list from the artikel navigator using the specified artikel ID.

    This function makes an authenticated GET request to the URL obtained from the 
    'artikel_navigator' entry in the 'urls' dictionary, appending the artikel ID 
    to the end of the URL and appending 'versies' after that. The credentials for authentication are retrieved from 
    environment variables.

    Args:
        artikel_id (str): The ID of the artikel to retrieve.

    Returns:
        The response from the GET request.
    """

    url = urls["artikel_navigator"]["url"] + artikel_id + "/versies"
    return requests.get(url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()

def retrieve_artikel_inhoud(artikel_id: str, artikel_versie: str):
    """
    Retrieves an artikel from the artikel navigator using the specified artikel ID and version.

    This function makes an authenticated GET request to the URL obtained from the 
    'artikel_navigator' entry in the 'urls' dictionary, appending the artikel ID 
    to the end of the URL and appending 'versies' after that. The credentials for authentication are retrieved from 
    environment variables.

    Args:
        artikel_id (str): The ID of the artikel to retrieve.
        artikel_versie (str): The version of the artikel to retrieve.

    Returns:
        The response from the GET request.
    """

    url = urls["artikel_navigator"]["url"] + artikel_id + "/versies/" + artikel_versie
    return requests.get(url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()

def retrieve_boek_from_artikel(boek_id: str):
    """
    Retrieves the book linked to the artikel using the specified artikel ID.

    Args:
        artikel_id (str): The ID of the artikel to retrieve.

    Returns:
        The response from the GET request.
    """

    url = urls["boek_navigator"]["url"] + boek_id
    return requests.get(url, auth=(os.getenv("NAVIGATOR_USERNAME"), os.getenv("NAVIGATOR_PASSWORD"))).json()

if __name__ == "__main__":
    print("Running file as main file. Testing API calls...")
    print(10 * '-')
    artikel = retrieve_artikel("44412")
    boek_id = artikel['metadata'][0]['href'].split('/')[-1]
    boek = VitoBoek(retrieve_boek_from_artikel(boek_id))
    boek.get_latest_versie()
    print(boek.get_boek_info())
    # print(clean_html(retrieve_artikel_inhoud('53475', versies[-1]['datumVanKracht'])['inhoud']))
