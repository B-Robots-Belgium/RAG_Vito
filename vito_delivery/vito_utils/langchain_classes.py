import uuid
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
from .vito_classes import VitoArticle
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

class WetsartikelReferentie(BaseModel):
    """
    Een referentie naar een specifiek wetsartikel dat in de tekst wordt genoemd.
    Voorbeeld: 'Art. 5, paragraaf 3 van het Milieuwetboek.'
    """
    artikel_nummer: Optional[str] = Field(
        ...,
        description="De ID of het nummer van het genoemde wetsartikel (bijv. 'Art. 5' of '53475')."
    )

class WetboekHeader(BaseModel):
    """
    Basismetadata voor een wettekst, zoals in welk wetboek deze staat, de editie,
    en de publicatiedatum.
    """
    naam_wetboek: Optional[str] = Field(
        ...,
        description="De naam of titel van het wetboek of de codex."
    )
    editie: Optional[str] = Field(
        ...,
        description="De editie of versie van het wetboek waarin deze tekst verschijnt."
    )
    publicatiedatum: Optional[str] = Field(
        ...,
        description="De publicatiedatum van deze wettekst."
    )

class WetboekFormulier(BaseModel):
    """
    Een gestructureerde weergave van een wettekst, inclusief verwijzingen
    naar andere wetsartikelen of codes.
    """
    referenties: List[WetsartikelReferentie] = Field(
        default_factory=list,
        description="Een lijst met alle in de tekst genoemde wetsartikelen."
    )
    samenvatting: Optional[str] = Field(
        ...,
        description="Een korte samenvatting of parafrase van de wettekst."
    )
    is_addendum: bool = Field(
        ...,
        description="Boolean die aangeeft of deze tekst een bijlage/extra is (True) of een eigenlijke wettekst (False)."
    )
    top_level: Optional[str] = Field(
        ...,
        description="This is the classification of the actual article. The classification is in Dutch and from the given options.",
        enum=['Handhaving', 'Stoffen', 'Compartiment']
    )

class Example(TypedDict):
    input: str
    tool_calls: List[BaseModel]

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into a language model."""
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    tool_calls = []
    
    for tool_call in example["tool_calls"]:
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "args": tool_call.dict(),
                "name": tool_call.__class__.__name__,
            },
        )
    
    messages.append(AIMessage(content="", tool_calls=tool_calls))
    
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(tool_calls)
    
    for output, tool_call in zip(tool_outputs, tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    
    return messages


def create_examples_and_messages() -> List[BaseMessage]:
    """
    Builds an example set of messages demonstrating how a user-supplied law-text 
    might be transformed into a structured object with references to other articles.
    """
    
    examples = [
        (
            "Volgens het Vlaamse Milieuwetboek (editie 2025) bepaalt artikel 53475 de reikwijdte van afvalbeheer. Daarnaast verduidelijkt Art. 12bis de beroepsprocedure. Deze tekst is gepubliceerd op 2025-02-10.",
            WetboekFormulier(
                referenties=[
                    WetsartikelReferentie(
                        artikel_nummer="art 5.3.4.7.5"
                    ),
                    WetsartikelReferentie(
                        artikel_nummer="12bis"
                    ),
                ],
                is_addendum=False,
                samenvatting=(
                    "Een wettekst over de reikwijdte van afvalbeheer en beroepsprocedures, "
                    "met verwijzing naar twee artikelen."
                ),
                top_level="Stoffen"
            )
        )
    ]

    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages(
                {"input": text, "tool_calls": [tool_call]}
            )
        )

    return messages