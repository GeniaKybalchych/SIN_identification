from data_preparer import create_training_data, check_entity_alignment
from evaluate import evaluate_model
from sin_model_trainer import SINModelTrainer

if __name__ == "__main__":
    """
    Main script for training and evaluating a Named Entity Recognition (NER) model 
    to identify Social Insurance Numbers (SIN) in text.
    """

    # Creating training data using a custom function
    training_data = create_training_data()
    for text, annot in training_data:
        # Display the training text and expected entities
        print(f"Text: {text}")
        print(f"Expected Entities: {annot['entities']}")

        # Checking entity alignment with BILOU tags
        biluo_tags = check_entity_alignment(text, annot['entities'])
        print(f"BILOU Tags: {biluo_tags}\n")

    # Initialize and train the SIN model
    trainer = SINModelTrainer()
    trainer.train_model(training_data)
    nlp_model = trainer.train_model(training_data)  # Train the model

    # Test the model on various sentences
    test_sentences = [
        "I just received my SIN number, it's 123456789.",
        "This is an example sentence without a SIN number.",
        "Her SIN, 987654321, was used for the application.",
        "No SIN number is mentioned in this particular sentence.",
        "Can you verify if 555555555 is a valid SIN?",
        "This sentence talks about 222222222 as a possible SIN number.",
        "Random text here but no Social Insurance Number.",
        "Is 444444444 the correct SIN for this account?",
        "SIN 777 777 777 was found to be invalid upon checking.",
        "He mentioned that his SIN, 888888888, needs updating.",
        "For identification purposes, your SIN 999999999 is required.",
        "SIN number 000 000 000 was not recognized by the system.",
        "According to our records, 321321321 is your SIN.",
        "This sentence is just a control sentence without numbers or SIN."
    ]
    for sentence in test_sentences:
        print(f"Testing on sentence: '{sentence}'")
        print(trainer.test_model(sentence))

    # Define test data for evaluation
test_data = [
    ("I just received my SIN number, it's 123456789.", {"entities": [(36, 45, "SIN")]}),
    ("This is an example sentence without a SIN number.", {"entities": []}),  # Pas de SIN
    ("Her SIN, 987654321, was used for the application.", {"entities": [(9, 18, "SIN")]}),
    ("No SIN number is mentioned in this particular sentence.", {"entities": []}),  # Pas de SIN
    ("Can you verify if 555555555 is a valid SIN?", {"entities": [(18, 27, "SIN")]}),
    ("This sentence talks about 222222222 as a possible SIN number.", {"entities": [(26, 35, "SIN")]}),
    ("Random text here but no Social Insurance Number.", {"entities": []}),  # Pas de SIN
    ("Is 444444444 the correct SIN for this account?", {"entities": [(3, 12, "SIN")]}),
    ("SIN 777 777 777 was found to be invalid upon checking.", {"entities": [(4, 15, "SIN")]}),
    ("He mentioned that his SIN, 888888888, needs updating.", {"entities": [(27, 36, "SIN")]}),
    ("For identification purposes, your SIN 999999999 is required.", {"entities": [(38, 47, "SIN")]}),
    ("SIN number 000 000 000 was not recognized by the system.", {"entities": [(11, 22, "SIN")]}),
    ("According to our records, 321321321 is your SIN.", {"entities": [(26, 35, "SIN")]}),
    ("This sentence is just a control sentence without numbers or SIN.", {"entities": []})  # Pas de SIN
]

# Process test data to check entity alignment
for text, annot in test_data:
    print(f"Text: {text}")
    print(f"Expected Entities: {annot['entities']}")
    biluo_tags = check_entity_alignment(text, annot['entities'])
    print(f"BILOU Tags: {biluo_tags}\n")

# Evaluate the model using the test data
precision, recall, f1_score = evaluate_model(nlp_model, test_data)

# Display evaluation results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
