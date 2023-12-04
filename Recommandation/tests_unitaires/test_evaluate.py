import unittest
from evaluate import evaluate_model
import spacy

class TestEvaluateModel(unittest.TestCase):
    """
    A test suite for the evaluate_model function.

    This class tests the evaluate_model function using different scenarios,
    including perfect predictions and cases with no correct predictions, to ensure
    the evaluation metrics (precision, recall, F1 score) are calculated correctly.
    """

    def setUp(self):
        """
        Set up method called before each test. Initializes the SpaCy model.
        """
        # Load the SpaCy model
        self.nlp = spacy.load("E:/test_assessement/my_model")

    def test_perfect_predictions(self):
        """
        Tests evaluate_model with perfect predictions to ensure it correctly calculates
        the metrics when all predictions are correct.
        """
        # Simulate perfect model predictions
        perfect_predictions = [
            ("My SIN number is 123456789", {"entities": [(17, 26, "SIN")]}),
        ]

        # Evaluate the model with perfect predictions
        precision, recall, f1_score = evaluate_model(self.nlp, perfect_predictions)

        # Check if precision, recall, and F1 score are perfect (1.0)
        self.assertEqual(precision, 1, "Precision should be 1 with perfect predictions")
        self.assertEqual(recall, 1, "Recall should be 1 with perfect predictions")
        self.assertEqual(f1_score, 1, "F1 score should be 1 with perfect predictions")

    def test_no_correct_predictions(self):
        """
        Tests evaluate_model with no correct predictions to ensure it correctly calculates
        the metrics when there are no correct predictions.
        """
        # Simulate a scenario with no correct predictions
        no_correct_predictions = [
            ("My SIN number is 123456789", {"entities": []}),  # No entity predicted
        ]

        # Evaluate the model with no correct predictions
        precision, recall, f1_score = evaluate_model(self.nlp, no_correct_predictions)

        # Check if precision, recall, and F1 score are zero
        self.assertEqual(precision, 0, "Precision should be 0 with no correct predictions")
        self.assertEqual(recall, 0, "Recall should be 0 with no correct predictions")
        self.assertEqual(f1_score, 0, "F1 score should be 0 with no correct predictions")

if __name__ == '__main__':
    unittest.main()