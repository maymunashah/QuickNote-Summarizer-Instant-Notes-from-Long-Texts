# QuickNote Summarizer: Instant Notes from Long Texts



## Features

**Model Evaluation:** Evaluate summarization models using ROUGE and BERTScore metrics on a subset of the CNN/DailyMail dataset.

**Interactive Inference:** Use the Gradio-powered web app to test summarization models with custom input.

## Project Structure
```bash
quicknote-summarizer/
├── src/
│   ├── evaluation.py     # Script to evaluate models and compute metrics
│   ├── app.py            # Gradio app for user interaction
│   └── results/          # Directory to store evaluation results
└── README.md             # Project documentation
```


## Prerequisites

Python 3.8+

pip or another package manager

Virtual environment (optional but recommended)

Installation
```bash
Clone the repository:

git clone <repository_url>
cd quicknote-summarizer
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure the necessary datasets and models are downloaded. The evaluation script will fetch the CNN/DailyMail dataset and the required transformer models automatically.

# Usage

## 1. Model Evaluation

Evaluate the summarization models and compute their ROUGE and BERTScore metrics.

Run the evaluation script:

python src/evaluation.py

This script:

Evaluates models on 1,000 random samples from the CNN/DailyMail dataset.

Computes ROUGE and BERTScore for each model.

Outputs detailed results in the src/results/ directory as a JSON and text file.

## 2. Interactive Inference with Gradio

Use the Gradio web app to test the summarization models with custom inputs.

Start the app:

python src/app.py

Open your browser at the provided local URL (e.g., http://127.0.0.1:7860/) to access the interface.

Enter text in the input box, and the app will generate a summary using the selected model.

## Evaluation Metrics

ROUGE: Measures the overlap of n-grams, capturing recall and precision of generated summaries compared to reference summaries.

BERTScore: Computes similarity using contextual embeddings, offering a more nuanced evaluation of summary quality.

## Results

Evaluation results are saved in the src/results/ directory:

evaluation_results.json: Contains detailed summaries, scores, and inference times for each model.

evaluation_results.txt: Summarized report with average, min, and max scores.

##  Future Enhancements

Add more models for evaluation and comparison.

Extend the Gradio app to include multi-lingual summarization support.

Improve evaluation with human feedback and additional datasets.

## Contributing

Contributions are welcome! If you find bugs or want to add features, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



