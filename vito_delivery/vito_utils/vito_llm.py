import psycopg2
import openai
import os
from .utils import clean_html
from .vito_classes import VitoArticle
from .langchain_classes import WetboekFormulier, create_examples_and_messages
from .db_actions import get_top_similar_items
from langchain.chat_models import init_chat_model 
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.callbacks import get_openai_callback

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

def classify_with_langchain(text: str, candidate_labels: list, ) -> str:
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        model_name="gpt-4o"
    )

    classification_prompt = PromptTemplate(
        input_variables=["text", "labels"],
        template="""
        You are given a piece of text:
        ---
        {text}
        ---

        You have these possible labels to choose from:
        {labels}

        Pick exactly one label from the above list that best fits the text, and output only that label (without explanation). 
                """,
            )

    # Build an LLMChain that uses the prompt above
    chain = classification_prompt | llm

    # Run the chain with text and candidate labels
    candidate_str = ", ".join(candidate_labels)
    llm_output = chain.invoke(
        {
            "text": text,
            "labels": candidate_str
        })
    print(llm_output)
    # The LLM output should be exactly one of the provided labels
    chosen_label = llm_output.text().strip()
    return chosen_label

def extract_with_langchain(article: VitoArticle) -> dict:
    LLM_client = init_chat_model(
        model="gpt-4-0125-preview",
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    ) 

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ""
            ),
            MessagesPlaceholder("examples"), 
            ("human", "{text}")
        ]
    )

    messages = create_examples_and_messages()

    # Create structured output handling
    runnable = prompt | LLM_client.with_structured_output(
        schema=WetboekFormulier,
        method="function_calling",
        include_raw=False,
    )
    # + "\n" + article.inhoud
    text = "Article name: " + article.artikel + "\n" + article.inhoud

    try:
        with get_openai_callback() as cb:
            response = runnable.invoke(
                {
                    "text": text,
                    "examples": messages
                }
            )

    except Exception as e:
        print(f"Error during extraction with Azure OpenAI: {e}")
        return None

    return response, cb