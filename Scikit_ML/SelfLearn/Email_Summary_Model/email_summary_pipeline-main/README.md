# Email Summary Model Web App

This is a complete end-to-end Machine Learning project demonstrating Extractive text summarization on emails. It perfectly mirrors the architecture of a Scikit-Learn Flask application.

## 🌟 Visuals & UI
We have designed a stunning **Glassmorphism** responsive UI using pure CSS (with dynamic gradient backgrounds and floating blobs) to ensure an ultra-premium experience when interacting with the Extractive Summarizer.

## 🛠️ Tech Stack
* **Python 3**
* **Flask** (for the web framework)
* **Scikit-Learn (TF-IDF)** & **SpaCy** (for Unsupervised Extractive Summarization)
* **Joblib** (for model serialization)
* **HTML/Vanilla CSS** (Frontend UI)

## 🚀 How to Run Locally

### 1. Install Dependencies
Open your terminal inside this project directory and install the necessary dependencies:
```bash
pip install -r requirements.email_summary_model.txt
```

### 2. Download the NLP Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Generate the ML Pipeline Model (Optional)
Run all the cells in the `email_summary_training.ipynb` Jupyter Notebook. 
This behaves just like `sentiment_model_training.ipynb`—it initializes our custom Scikit-Learn `ExtractiveSummarizer` class, tests it, and saves a `email_summary_model.pkl` file using Joblib.

### 4. Start the Application
```bash
python app.py
```

### 5. Start Summarizing
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

Enjoy!
