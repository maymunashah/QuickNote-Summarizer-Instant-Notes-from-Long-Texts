

import gradio as gr
from src.summarize import Summarizer  # Import your Summarizer class


# Create an instance of the Summarizer class
summarizer = Summarizer()

# Function to call the summarize method and get output
def summarize_text(text):
    summary = summarizer.summarize(text)  # Use the default method of summarization
    return summary

# Define Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to summarize..."),
    outputs=gr.Textbox(label="Summary"),
    title="QuickNote Summarizer",
    description="Summarize long notes or text with a click of a button.",
    theme="huggingface",  # Optional: You can customize the theme
)

# Launch the interface
iface.launch()
