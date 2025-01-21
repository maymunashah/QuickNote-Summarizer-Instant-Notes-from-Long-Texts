

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import unittest
from src.summarize import Summarizer

class TestSummarizer(unittest.TestCase):

    def setUp(self):
        # Setup the Summarizer object
        self.summarizer = Summarizer()
        self.test_text = (
            "In the heart of the city, a bustling marketplace exists. Vendors sell fruits, vegetables, "
            "and fresh produce from the region. The atmosphere is filled with chatter, laughter, and the "
            "sounds of goods being exchanged. People from all walks of life come here to shop, meet friends, "
            "and enjoy the vibrant energy of the market."
        )

    def test_summarization(self):
        # Test the summarizer method
        summary = self.summarizer.summarize(self.test_text)
        print(summary)
        self.assertTrue(len(summary) > 0, "Summary should not be empty.")

if __name__ == "__main__":
    unittest.main()
