

from src.preprocess import Preprocessor
from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Initialize the Hugging Face summarization pipeline
        self.summarizer = pipeline("summarization")
        self.preprocessor = Preprocessor()

    def summarize(self, text):
        # Preprocess the input text
        processed_text = self.preprocessor.preprocess(text)

        # Generate the summary using the Hugging Face pipeline
        summary = self.summarizer(processed_text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
