import spacy
from spacy.training import offsets_to_biluo_tags


def create_training_data():
    """
    Create training data for Named Entity Recognition (NER) using SpaCy format.

    Returns:
        list of tuple: A list of text and entity annotations.
    """
    return [
        ("My SIN number is 123 456 789.", {"entities": [(17, 28, "SIN")]}),
        ("My SIN number is 345123890.", {"entities": [(17, 26, "SIN")]}),
        ("She said her SIN was 987654321.", {"entities": [(21, 30, "SIN")]}),
        ("I found a SIN: 567321456 in the document.", {"entities": [(15, 24, "SIN")]}),
        ("Remember to update your 345258111.", {"entities": [(24, 33, "SIN")]}),
        ("He noted the number 333 746 210.", {"entities": [(20, 31, "SIN")]}),
        ("His SIN, 635 560 310, was incorrect.", {"entities": [(9, 20, "SIN")]}),
        ("Can you check the 490 301 319?", {"entities": [(18, 29, "SIN")]}),
        ("I lost my wallet with my 865 902 814.", {"entities": [(25, 36, "SIN")]}),
        ("Insurance number: 690024836.", {"entities": [(18, 27, "SIN")]}),
        ("Here is my new SIN: 107734023.", {"entities": [(20, 29, "SIN")]}),
        ("For the record, use 578 823 012.", {"entities": [(20, 31, "SIN")]}),
    ]


def check_entity_alignment(text, entities):
    """
    Check alignment of entities in text and generate BILOU tags.

    Args:
        text (str): The input text.
        entities (list of tuple): List of entity annotations as (start, end, label).

    Returns:
        list of str: BILOU tags indicating entity alignment in the text.
    """
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    biluo_tags = offsets_to_biluo_tags(doc, entities)
    return biluo_tags
