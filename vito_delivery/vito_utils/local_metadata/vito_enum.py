from enum import Enum

class Thema(Enum):
    algemeen = 1
    leefmilieu = 2
    natuur = 3
    energie = 4
    ruimtelijkeOrdening = 5
    dierenwelzijn = 6

class Type(Enum):
    wetten = 1
    decreten = 2
    besluitenVlaamseRegering = 3
    koninklijkeBesluiten = 4
    ministerieleBesluiten = 5
    omzendbrieven = 6
    samenwerkingsakkoorden = 7
    europeseVerordeningen = 8
    europeseRichtlijnen = 9
    internationaleVerdragen = 10
    huishoudelijkeReglementen = 11
    europeseUitvoeringsbesluiten = 13
    europeseUitvoeringsverordeningen = 14

class Toepassingsgebied(Enum):
    vlaams = 1
    federaal = 2
    europees = 3
    internationaal = 4