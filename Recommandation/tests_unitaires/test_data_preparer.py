import unittest
from data_preparer import create_training_data, check_entity_alignment

class TestDataPreparer(unittest.TestCase):
    """
    A test suite for the data preparation functionalities in the data_preparer module.

    This class tests both the create_training_data function, which is supposed to generate
    a list of tuples with text and annotations, and the check_entity_alignment function,
    which checks for the correct BILOU tagging of identified entities.
    """

    def test_create_training_data(self):
        """
        Tests the create_training_data function to ensure it returns data in the correct format and content.
        """
        # Call the function to get the training data
        data = create_training_data()

        # Check that data is a list
        self.assertIsInstance(data, list, "The function should return a list")

        for item in data:
            # Check that each item in the list is a tuple
            self.assertIsInstance(item, tuple, "Each item in the list should be a tuple")

            # Check that the tuple contains two elements: text and annotations
            self.assertEqual(len(item), 2, "Each tuple should have two elements")

            # Check that the first element of the tuple is a string (the text)
            self.assertIsInstance(item[0], str, "The first element of the tuple should be a string")

            # Check that the second element of the tuple is a dictionary (the annotations)
            self.assertIsInstance(item[1], dict, "The second element of the tuple should be a dictionary")

            # Check that the dictionary contains the key 'entities'
            self.assertIn('entities', item[1], "The dictionary should contain the key 'entities'")

            # Check that the value associated with the key 'entities' is a list
            self.assertIsInstance(item[1]['entities'], list, "The value associated with 'entities' should be a list")

            # Verify each entity annotation in the 'entities' list
            for start, end, label in item[1]['entities']:
                # Verify that the label is 'SIN'
                self.assertEqual(label, 'SIN', "The label of the entities should be 'SIN'")

                # Extract the substring of the text corresponding to the entity indices
                entity_text = item[0][start:end]

                # Verify that the substring corresponds to a plausible format of SIN
                self.assertTrue(entity_text.replace(' ', '').isdigit(), "The extracted entity should be numeric")

                # Ensure that the indices do not exceed the length of the text
                self.assertTrue(end <= len(item[0]), "The end index should not exceed the length of the text")

    def test_check_entity_alignment(self):
        """
        Tests the check_entity_alignment function to ensure it returns the correct BILOU tags.
        """
        # Test cases with different sentence structures and entity positions
        cases = [
            ("My SIN number is 123456789.", {"entities": [(17, 26, "SIN")]}, ['O', 'O', 'O', 'O', 'U-SIN', 'O']),
            ("123456789 is my SIN number.", {"entities": [(0, 9, "SIN")]}, ['U-SIN', 'O', 'O', 'O', 'O', 'O']),
            ("My SIN number is 987654321.", {"entities": [(17, 26, "SIN")]}, ['O', 'O', 'O', 'O', 'U-SIN', 'O']),
            ("This is an example sentence without a SIN number.", {"entities": []}, ['O'] * 10)
        ]

        for text, entities, expected_tags in cases:
            # Assert that the returned BILOU tags match the expected tags
            biluo_tags = check_entity_alignment(text, entities["entities"])
            self.assertEqual(biluo_tags, expected_tags,
                             f"BILOU tags should match the expected pattern for the test case: '{text}'.")

if __name__ == '__main__':
    unittest.main()
