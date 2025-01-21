import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK datasets
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Preprocessor:
    """
    A class to handle text preprocessing tasks.
    """

    def __init__(self):
        """Initialize any required preprocessing resources."""
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        """
        Clean input text by removing special characters and converting it to lowercase.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        return text

    def tokenize_and_remove_stopwords(self, text):
        """
        Tokenize the text and remove stopwords.

        Args:
            text (str): Input text to tokenize.

        Returns:
            list: List of tokenized words without stopwords.
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return filtered_words

    def preprocess(self, text):
        """
        Combine the cleaning and tokenizing steps to preprocess the text.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        cleaned_text = self.clean_text(text)
        processed_text = self.tokenize_and_remove_stopwords(cleaned_text)
        return " ".join(processed_text)
 
