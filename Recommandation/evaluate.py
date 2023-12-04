def evaluate_model(nlp, test_data):
    """
    Evaluates the performance of an NLP model on a test dataset.

    Calculates precision, recall, and F1-score by comparing the entities predicted
    by the model with the actual (true) entities in the test data.

    Parameters:
    nlp : The trained NLP model for entity recognition.
    test_data : A dataset for testing, where each item is a tuple containing
                a text and its entity annotations.

    Returns:
    A tuple containing the precision, recall, and F1-score of the model.
    """

    # Initialize counters for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for text, annotations in test_data:
        # Process the text with the model
        doc = nlp(text)
        # Extract predicted entities
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        # Retrieve actual entities from annotations
        true_entities = annotations['entities']

        # Convert entities to sets for easy comparison
        true_entities = {(start, end, label) for start, end, label in true_entities}
        predicted_entities = {(start, end, label) for start, end, label in predicted_entities}

        # Update counters for true positives and false positives
        for entity in predicted_entities:
            if entity in true_entities:
                true_positives += 1
            else:
                false_positives += 1

        # Update counter for false negatives
        for entity in true_entities:
            if entity not in predicted_entities:
                false_negatives += 1

    # Calculate precision and recall, handling to avoid division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score