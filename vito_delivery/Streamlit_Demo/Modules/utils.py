# Modules/utils.py

import re
import os
import csv
from io import StringIO
from openai import OpenAI

def UploadOpenAI(text: str, client):
    response = client.chat.completions.create(
        model = "gpt-4o",
        response_format= {"type": "json_object"},
        messages = [
            {"role": "system", "content": "You are generating metadata for the following article. Always return the found information in JSON format"},
            {"role": "system", "content": "You will always return the information in the following JSON format, with Keyword being a general filler and the 'xxx' being the extraxted keyword:"},
            {"role": "system", "content": "{'metadata': [ {'Keyword': 'xxx'}, {'Keyword': 'xxx'} ]}"},
            {"role": "system", "content": "The metadata should be keywords that describe the article and are found within the article. Preferably these are keywords that can be used for clustering techniques. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "Extract only the 10 most relevant keywords from the following article. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "All keywords returned should be in Dutch. The text you will receive is also in Dutch."},
            {"role": "system", "content": "There should be no specific measurements. Always return the generated metadata in JSON format"},
            {"role": "system", "content": "The following message holds the text from which to generate the the metadata. Always return the generated metadata in JSON format"},
            {"role": "user", "content": text}
        ])
    return response

def parse_pg_array(array_str):
    trimmed = array_str.strip('{}')
    f = StringIO(trimmed)
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        return [item.strip() for item in row]

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
    else:
        return 1.0, 1.0