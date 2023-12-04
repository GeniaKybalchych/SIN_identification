import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random


class SINModelTrainer:
    """
    A class for training a Named Entity Recognition (NER) model in SpaCy
    specifically for recognizing SIN (Social Insurance Number) entities.
    """

    def __init__(self):
        """
        Initializes the SINModelTrainer instance by creating a blank English language model.
        """
        # Initialize the SpaCy model for the English language
        self.nlp = spacy.blank("en")

    def train_model(self, training_data, save_path):
        """
        Trains the NER model using the provided training data.

        Parameters:
        training_data (list): A list of tuples, each containing a text and its annotations.

        Returns:
        A trained SpaCy model.
        """
        # Convert training data into SpaCy Example objects
        examples = [Example.from_dict(self.nlp.make_doc(text), annotations)
                    for text, annotations in training_data]

        # Set up the NER pipeline
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")

        # Add the custom label 'SIN' for our specific entity
        ner.add_label("SIN")

        # Disable other pipelines during training to focus on NER
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.initialize()

            # Training loop
            for epoch in range(50):  # Number of training iterations
                random.shuffle(examples)  # Shuffle training data for each epoch
                for batch in minibatch(examples, size=compounding(4., 32., 1.001)):
                    # Update the model
                    self.nlp.update(batch, drop=0.5, sgd=optimizer)

            # Save the trained model
            self.nlp.to_disk(save_path)

        return self.nlp

    def test_model(self, text):
        """
        Tests the trained model on a given text.

        Parameters:
        text (str): The text to be processed by the trained NER model.

        Returns:
        List of tuples containing the entities recognized and their labels.
        """
        # Process the text through the trained model
        doc = self.nlp(text)

        # Return identified entities and their labels
        return [(ent.text, ent.label_) for ent in doc.ents]
