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
    artikel_id: Optional[str] = Field(
        ...,
        description="De ID of het nummer van het genoemde wetsartikel (bijv. 'Art. 5' of '53475')."
    )
    artikel_tekst: Optional[str] = Field(
        ...,
        description="De letterlijke tekstpassage waarin dit wetsartikel wordt vermeld."
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
    header: Optional[WetboekHeader] = Field(
        ...,
        description="De basisinformatie over deze wettekst."
    )
    referenties: List[WetsartikelReferentie] = Field(
        default_factory=list,
        description="Een lijst met alle in de tekst genoemde wetsartikelen."
    )
    samenvatting: Optional[str] = Field(
        ...,
        description="Een korte samenvatting of parafrase van de wettekst."
    )

class Example(TypedDict):
    input: str
    tool_calls: List[BaseModel]

def tool_example_to_messages(input_text: str, model_obj: BaseModel) -> List[BaseMessage]:
    """
    Convert a text string plus a Pydantic model into a list of messages
    that can be fed into a language model via LangChain.

    The idea is:
      1. The user provides text (HumanMessage).
      2. The AI “calls a tool” by returning a structured object (AIMessage with tool_calls).
      3. We then optionally add a ToolMessage to simulate a response from that tool.
    """
    messages: List[BaseMessage] = [HumanMessage(content=input_text)]

    tool_call_id = str(uuid.uuid4())
    messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "args": model_obj.dict(),
                    "name": model_obj.__class__.__name__,
                }
            ],
        )
    )

    messages.append(
        ToolMessage(
            content="Tool successfully received the structured data.",
            tool_call_id=tool_call_id,
        )
    )

    return messages


def create_examples_and_messages() -> List[BaseMessage]:
    """
    Builds an example set of messages demonstrating how a user-supplied law-text 
    might be transformed into a structured object with references to other articles.
    """
    example_law_text = (
    "Volgens het Vlaamse Milieuwetboek (editie 2025) bepaalt artikel 53475 de reikwijdte van afvalbeheer. "
    "Daarnaast verduidelijkt Art. 12bis de beroepsprocedure. Deze tekst is gepubliceerd op 2025-02-10."
)

    law_document_example = WetboekFormulier(
        header=WetboekHeader(
            naam_wetboek="VLAREM",
            editie="2025",
            publicatiedatum="2025-02-10"
        ),
        referenties=[
            WetsartikelReferentie(
                artikel_id="53475",
                artikel_tekst="artikel 53475 bepaalt de reikwijdte van afvalbeheer"
            ),
            WetsartikelReferentie(
                artikel_id="12bis",
                artikel_tekst="Art. 12bis verduidelijkt de beroepsprocedure"
            ),
        ],
        samenvatting=(
            "Een wettekst over de reikwijdte van afvalbeheer en beroepsprocedures, "
            "met verwijzing naar twee artikelen."
        )
    )

    example_messages = tool_example_to_messages(
        input_text=example_law_text,
        model_obj=law_document_example
    )

    return example_messages