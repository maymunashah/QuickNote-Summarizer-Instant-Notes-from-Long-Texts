from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Initialize the Hugging Face summarization pipeline with facebook/bart-large-cnn
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=-1,  # Use CPU
        )

    def summarize(self, text):
        # Check the tokenized length of the input text
        max_input_length = 1024  # Maximum input tokens for BART
        
        # Tokenize and truncate the input if necessary
        inputs = self.summarizer.tokenizer(text, max_length=max_input_length, truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        
        if len(input_ids) > max_input_length:
            # Split the input into two parts if it exceeds the max length
            midpoint = len(input_ids) // 2
            part1_ids = input_ids[:midpoint]
            part2_ids = input_ids[midpoint:]
            
            # Decode back to text
            part1 = self.summarizer.tokenizer.decode(part1_ids, skip_special_tokens=True)
            part2 = self.summarizer.tokenizer.decode(part2_ids, skip_special_tokens=True)

            # Summarize each part separately
            summary1 = self.summarizer(part1, min_length=50, do_sample=False)[0]['summary_text']
            summary2 = self.summarizer(part2, min_length=50, do_sample=False)[0]['summary_text']
            
            # Combine summaries
            combined_summary = f"{summary1} {summary2}"
        else:
            # Directly summarize if input length is within the limit
            print("no manual truncation was required")
            combined_summary = self.summarizer(text, min_length=50, do_sample=False)[0]['summary_text']

        return combined_summary

# Example Usage
if __name__ == "__main__":
    summarizer = Summarizer()
    text = """Your long input text goes here..."""
    summary = summarizer.summarize(text)
    print("Summary:", summary)


# from src.preprocess import Preprocessor
# from transformers import pipeline

# class Summarizer:
#     def __init__(self):
#         # Initialize the Hugging Face summarization pipeline
#         self.summarizer = pipeline("summarization")
#         self.preprocessor = Preprocessor()

#     def summarize(self, text):
#         # Preprocess the input text
#         processed_text = self.preprocessor.preprocess(text)

#         # Generate the summary using the Hugging Face pipeline
#         summary = self.summarizer(processed_text, max_length=150, min_length=50, do_sample=False)
#         return summary[0]['summary_text']
