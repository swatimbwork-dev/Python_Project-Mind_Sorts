# Python_Project-Mind_Sorts
Mind_Sorts - Text Classifier for Mental Health Signals
## Project Overview
Mind-Sorts is an NLP-based text classification application that classifies mental-health-related text into four predefined categories:
* Anxiety
* Depression
* Normal
* Suicidal
## Approach
The application uses the following machine learning pipeline:
**Text → Word + Character TF-IDF → Logistic Regression → Predicted Category**
TF-IDF converts text into numerical features, and Logistic Regression uses these features to predict the category.
## How to Use
### 1 Website: https://swatimbwork-mind-sorts.hf.space/
1. Enter a sentence in the text box or Select from "Try an illustrative Example"
2. Click **Classify Text**.
3. View the predicted category and category scores.
### 2 Local 
#### 2.1 Install dependencies
Open Anaconda Prompt or a terminal in this folder and run:
```bash
python -m pip install -r requirements.txt
```
#### 2.1 Run the app locally
```bash
python app.py
```
Open the local URL shown in the terminal. You can stop the app with `Ctrl+C`.
http://127.0.0.1:7860
## Model Evaluation
The final model achieved the following results on the held-out test dataset:
* Accuracy: 81.65%
* Macro F1-score: 81.24%
Depression was the most challenging category to classify, with an F1-score of 71%.
## Limitations
The model classifies text into predefined categories and may confuse categories that use similar language. Its performance depends on the training dataset and the text provided.

**Disclaimer:** This application is for educational and research demonstration purposes only. It is not a clinical diagnostic tool and should not be used to make medical decisions.
