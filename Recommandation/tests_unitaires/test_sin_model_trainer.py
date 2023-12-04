import unittest
import os
from sin_model_trainer import SINModelTrainer
from data_preparer import create_training_data

class TestSINModelTrainer(unittest.TestCase):
    """
    A test suite for the SINModelTrainer class in the sin_model_trainer module.

    This class includes tests for checking the initialization of the SINModelTrainer
    and the successful execution of its training process.
    """

    def test_initialization(self):
        """
        Tests the initialization of the SINModelTrainer to ensure that the NLP model is correctly initialized.
        """
        # Initialize the SINModelTrainer
        trainer = SINModelTrainer()

        # Check that the NLP model within the trainer is initialized
        self.assertIsNotNone(trainer.nlp, "NLP model should be initialized")

    def test_training_process(self):
        """
        Tests the training process of the SINModelTrainer to ensure that it successfully updates the model's NER pipeline.
        """
        # Initialize the SINModelTrainer
        trainer = SINModelTrainer()
        # Create training data
        training_data = create_training_data()

        # Run the training process
        trainer.train_model(training_data)

        # Check if the model's 'ner' pipeline has been updated after training
        self.assertIn('ner', trainer.nlp.pipe_names, "NER pipeline should exist in the model after training")

if __name__ == '__main__':
    unittest.main()
