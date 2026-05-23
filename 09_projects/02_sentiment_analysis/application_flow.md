# Sentiment Analysis Application Flow

This document outlines the end-to-end data flow of the Sentiment Analysis web application, tracking what happens from the user's initial interaction on the UI all the way to the machine learning model's prediction and the final output rendering.

## 1. Application Startup (`app.py`)
Before the user even interacts with the UI, the Flask backend needs to be initialized and the machine learning model needs to be loaded into memory.

When you run `python app.py`:
- The Flask web server is initialized.
- The pre-trained Scikit-Learn pipeline (`sentiment_model.pkl`) is loaded using `joblib`.
- This loaded pipeline consists of:
  - Default preprocessing steps.
  - A `TfidfVectorizer` which utilizes the custom tokenization logic defined in the `CustomTokenizer` class (from `custom_tokenizer_function.py`).
  - A `LinearSVC` Classifier containing the trained model weights.

---

## 2. Serving the UI (Targeting `/`)
- **User Action:** The user opens a web browser and navigates to the application's URL (e.g., `http://127.0.0.1:5000/`).
- **Backend Routing:** The request hits the root route defined in `app.py` (`@app.route('/')`).
- **Rendering:** The `home()` function is executed. It uses `render_template('index.html')` to serve the main HTML user interface.

---

## 3. The HTML Form (`templates/index.html`)
The user is presented with the UI rendered from `index.html`. It contains a form configured for user input.

```html
<form action="{{ url_for('predict')}}" method="post">
    <!-- Input Field -->
    <input type="text" name="Enter the product review here" required="required" />
    
    <!-- Submit Button -->
    <button type="submit">Predict Sentiment</button>
</form>
```

- **User Action:** The user types an Amazon Alexa Review (e.g., `"I love my new Alexa!"`) into the text box and clicks the **Predict Sentiment** button.
- **Data Transfer:** Clicking the submit button triggers an HTTP `POST` request. The browser bundles the input values and sends this request to the designated form action, which maps to the `/predict` route in the backend.

---

## 4. Backend Processing (`app.py` - `/predict` Route)
The Flask application receives the `POST` request and routes it to the `predict()` function.

- **Data Extraction:** The backend unpackages the user's input from the HTML form request object:
  ```python
  new_review = [str(x) for x in request.form.values()]
  ```
  This creates a list containing the user's review as a string (`["I love my new Alexa!"]`), as the underlying Scikit-Learn `model.predict()` method expects a list/array-like input.

---

## 5. Machine Learning Pipeline Execution
The text is passed sequentially through the loaded pipeline:
```python
prediction = model.predict(new_review)[0]
```

This single line actually triggers a series of complex operations:

### A. Tokenization & Cleaning (`custom_tokenizer_function.py`)
Because the pipeline's vectorizer was configured with the `CustomTokenizer`, the raw string is immediately passed to its `text_data_cleaning` method.
- **spaCy Tokenization:** The text is parsed by `en_core_web_sm` into words.
- **Lemmatization:** Words are converted to their root forms (e.g., "loved" becomes "love").
- **Stopword/Punctuation Removal:** Words that don't add semantic value (like "I" or "my") and punctuation are stripped.
- *Output:* The sentence `"I love my new Alexa!"` becomes `['love', 'new', 'alexa']`.

### B. TF-IDF Vectorization
The pipeline's `TfidfVectorizer` takes the cleaned tokens and transforms them into a numerical vector based on the vocabulary it learned during training.

### C. the Support Vector Classifier (`LinearSVC`)
The numerical vector is fed into the Support Vector Classifier. The model calculates the decision boundary and predicts whether the sentiment is `0` (Negative) or `1` (Positive).

---

## 6. Output Generation & UI Rerender (`app.py` & `index.html`)
The output value from the model prediction is stored in the `prediction` variable.

- **Condition Logic:** Inside the `predict()` route, an `if/else` statement checks this value:
  ```python
  if prediction == 0:
      return render_template('index.html', prediction_text='Negative 👎')
  else:
      return render_template('index.html', prediction_text='Positive 👍')
  ```

- **UI Update:** The server renders the same `index.html` file, but this time it dynamically passes the `prediction_text` variable to the Jinja templating engine.
- **Final Display:** The variable is injected into the HTML where `{{ prediction_text }}` is located, successfully displaying "Positive 👍" or "Negative 👎" on the user's screen.
