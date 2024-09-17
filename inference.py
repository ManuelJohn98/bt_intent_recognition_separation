import os
from pprint import pprint
from transformers import pipeline
from config import model_directory

classifier = pipeline(
    "ner",
    os.path.join(
        model_directory,
        "ikim-uk-essen/geberta-base_intent_recognition_separation/checkpoint-750",
    ),
    device=0,
)

pprint(classifier("Hier Gruppenführer 1. Wie lange wird der Einsatz dauern?"))
