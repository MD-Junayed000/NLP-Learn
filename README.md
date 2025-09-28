# NLP Learning Lab

This repository is a personal playground for natural language processing (NLP) experiments. It mixes quick Python scripts, exploratory Jupyter notebooks, and curated datasets that walk through the NLP stack: text cleaning, classical feature engineering, neural embeddings, sequence models, transformers, and downstream tasks such as fake-news and spam detection.

## Whats Inside
- Core scripts (`*.py`) - compact examples that you can run from the terminal to see the fundamentals in action:
  - `Tokenization.py` - sentence and word tokenisation with NLTK.
  - `stemming.py` - stop-word removal plus Porter stemming.
  - `lemmining.py` - WordNet lemmatisation (name kept to honour its origin).
  - `BagOFWords.py` - classical bag-of-words pipeline with `CountVectorizer`.
  - `TF-IDF.py` - TF-IDF features over the A. P. J. Abdul Kalam speech snippet.
  - `Word2vec.py` - training a tiny Word2Vec model with Gensim.
- Jupyter notebooks for deeper dives and modelling case studies:
  - `bert_intro.ipynb`, `BERT_email_classification-handle-imbalance.ipynb` - transformer fine-tuning walkthroughs.
  - `Kaggle Faker News Classifier Using LSTM- Deep LEarning.ipynb`, `FAKE NEWS CLASSIFIER .ipynb` - recurrent and deep learning baselines for misinformation detection.
  - `Stacked LSTM-Forecasting and Stock Price Prediction .ipynb`, `Stock Price Movement Based On News Headline using NLP.ipynb` - combining NLP with time-series forecasting.
  - `Regex For NLP.ipynb`, `Regular expression.ipynb`, `Learning Gradient Descent .ipynb`, and similar primers on fundamentals.
- Datasets and assets - reusable CSV or TXT files such as `SMSSpamCollection.txt`, `Stock-Sentiment-Analysis.csv`, the `fake-news/` splits, and reference documents (`NLP.docx`, `NLP code explain.pdf`).
- Support files - `.venv/` for local virtual environment artefacts, `.ipynb_checkpoints/` for notebook autosaves, and `.idea/` for JetBrains settings.

## Getting Started
1. Python environment
   ```powershell
   cd C:\Users\ASUS\nlp
   python -m venv .venv
   .\.venv\Scripts\Activate
   python -m pip install --upgrade pip
   pip install nltk scikit-learn gensim pandas numpy matplotlib seaborn jupyter notebook
   # Optional (needed for deep-learning notebooks)
   pip install tensorflow keras torch torchvision torchaudio transformers
   ```
2. Download required NLTK data once
   ```powershell
   python - <<'PY'
   import nltk
   for pkg in ["punkt", "stopwords", "wordnet"]:
       nltk.download(pkg)
   PY
   ```
3. Launch Jupyter for notebooks
   ```powershell
   jupyter notebook
   ```

## Running the Quick Scripts
All core scripts are self-contained. Activate your virtual environment, then execute for a fast demo:
```powershell
python Tokenization.py
python stemming.py
python BagOFWords.py
python TF-IDF.py
python Word2vec.py
```
Each script prints intermediate representations (cleaned corpus, matrices, learned vocabulary, and so on) to help you verify the processing steps. Update the sample paragraphs with your own text to experiment further.

## Working With Notebooks
- Every notebook is standalone; open it in Jupyter or VS Code and run the cells sequentially.
- Heavy notebooks (BERT, LSTM) expect a GPU-enabled environment or patience on CPU.
- Keep an eye on dataset paths. Large files such as `fake-news/train.csv` or `Stock-Sentiment-Analysis.csv` live in the repository root or `fake-news/`. Adjust relative paths if you move notebooks around.

## Data Notes
- Datasets are included for reproducibility and quick experimentation. Verify their licences before redistributing.
- Many CSVs are large (some exceed 50 MB). Git operations may feel slow; consider Git LFS if you plan to collaborate.
- `AAPL.csv` and other financial files pair with the stock forecasting notebooks. `SMSSpamCollection.txt` supports classic spam detection examples.

## Extending the Playground
- Add a `requirements.txt` if you want to lock dependencies.
- Swap the placeholder paragraphs in the scripts for domain-specific corpora to see how tokenisation and vectorisers behave.
- Try modern alternatives: `spaCy` for preprocessing, `sentence-transformers` for embeddings, or `langchain` for chaining workflows.
- Convert notebooks that you use frequently into Python packages or CLI tools for repeatable pipelines.

## Troubleshooting
- Import errors: double-check your virtual environment and installed packages.
- NLTK lookup errors: re-run the downloader snippet above or set the `NLTK_DATA` environment variable to point to your corpus directory.
- Encoding issues on Windows: prefer UTF-8 when loading or saving new datasets (`encoding="utf-8"`).

Happy experimenting! Document new learnings in the notebooks or expand this README as the collection grows.
