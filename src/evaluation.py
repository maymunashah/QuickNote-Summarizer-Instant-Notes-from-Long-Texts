import time
import torch
import gc
from transformers import pipeline, BartForConditionalGeneration, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import pandas as pd
from tqdm import tqdm
import os

# Function to evaluate model performance on CPU and ROUGE score
def evaluate_models_on_cpu():
    # Load the dataset (CNN/DailyMail)
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')  # Entire test set
    
    # Shuffle and select 1000 random samples from the dataset
    dataset = dataset.shuffle(seed=42)  # Setting a seed for reproducibility
    test_texts = dataset['article'][:100]  # (replace 100 by smaller number if you would like to test on a smaller part of test set)
    # Initialize results list
    results = []
    
    # Load ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Measure performance (latency) and summarize the text
    def summarize_with_timing(summarizer, text):
        start_time = time.time()
        summary = summarizer(text)
        end_time = time.time()
        return summary[0]['summary_text'], end_time - start_time

    # List of models to evaluate
    model_info = [
        {
            "model_name": "facebook/bart-large-cnn",
            "model_class": BartForConditionalGeneration,
            "max_length": 1024,
        },
        {
            "model_name": "t5-large",
            "model_class": T5ForConditionalGeneration,
            "max_length": 512,
        },
        {
            "model_name": "Falconsai/text_summarization",
            "model_class": AutoModelForSeq2SeqLM,
            "max_length": 512,
        },
        {
            "model_name": "google/pegasus-xsum",
            "model_class": AutoModelForSeq2SeqLM,
            "max_length": 512,
        }
    ]
    
    for model in tqdm(model_info):
        model_name = model["model_name"]
        model_class = model["model_class"]
        max_length = model["max_length"]
        
        print(f"Loading model {model_name}...")
        # Load the model and tokenizer
        model_instance = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Adjust max_length based on model's configuration
        max_length = getattr(model_instance.config, "max_position_embeddings", max_length)
        
        summarizer = pipeline(
            'summarization',
            model=model_instance,
            tokenizer=tokenizer,
            device=-1,
            max_length=200,  # Maximum output tokens
            truncation=True  # Automatically truncate inputs to model's max length
        )

        for text in tqdm(test_texts):
            print(f"Original text length: {len(text.split())}")
            
            # Tokenize and truncate input
            inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
            text_stripped = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            print(f"Truncated text length: {len(text_stripped.split())}")
            
            summary, elapsed_time = summarize_with_timing(summarizer, text_stripped)
            print(summary)
            print("_*************************************")

            rouge_score = rouge.compute(predictions=[summary], references=[dataset['highlights'][test_texts.index(text)]])
            print(rouge_score['rouge1'])
            # Calculate word counts
            original_word_count = len(text.split())
            summary_word_count = len(summary.split())

            # Store the results in the list
            results.append({
                'original_text': text,
                'summarized_text': summary,
                'model_name': model_name,
                'rouge1_f1': rouge_score['rouge1'],
                'rouge2_f1': rouge_score['rouge2'],
                'rougeL_f1': rouge_score['rougeL'],
                'rougeLsum_f1': rouge_score['rougeLsum'],
                'rouge_avg' : (rouge_score['rouge1']+rouge_score['rouge2']+rouge_score['rougeL']+rouge_score['rougeLsum'])/4,
                'response_time': elapsed_time,
                'original_text_word_count': original_word_count,
                'summarized_text_word_count': summary_word_count,
            })

        # After evaluating this model, clear cache to free up memory
        del model_instance
        del summarizer
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU memory if CUDA is being used
        
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save the DataFrame to CSV for analysis
    output_dir = r"summariserAI\quicknote-summarizer\src\results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_results.to_csv(r'D:\projects\summariserAI\quicknote-summarizer\src\results\evaluation_results_v3.csv', index=False)
    
    return df_results

# Run the evaluation and print the results
df_results = evaluate_models_on_cpu()
print("\nDataframe of Evaluation Results:")
print(df_results.head())  # Display the first few rows of the results
