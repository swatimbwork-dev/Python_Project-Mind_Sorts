from pathlib import Path
import joblib
import gradio as gr
from scipy.sparse import hstack

MODEL_PATH = Path(__file__).with_name("mind_sorts_model.joblib")

EXAMPLES = [
    ["I feel nervous and worried about everything lately."],
    ["I have lost interest in things I used to enjoy."],
    ["I had a great day and feel happy and relaxed."],
    ["I feel hopeless and don't know how to cope."],
]


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}.\n"
            "Export your trained model from Evaluation.ipynb first. "
            "See README.md for instructions."
        )
    bundle = joblib.load(MODEL_PATH)
    required = {"word_tfidf", "char_tfidf", "model"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(
            "The model file is missing these required items: " + ", ".join(sorted(missing))
        )
    return bundle


bundle = None
load_error = None
try:
    bundle = load_model_bundle()
except Exception as exc:
    load_error = str(exc)


def classify_text(text):
    if not text or not text.strip():
        return {}, "Please enter some text first."

    if bundle is None:
        return {}, (
            "The trained model has not been added yet. Export `mind_sorts_model.joblib` "
            "from your notebook and place it beside `app.py`.\n\nDetails: " + str(load_error)
        )

    clean_text = text.strip()
    word_features = bundle["word_tfidf"].transform([clean_text])
    char_features = bundle["char_tfidf"].transform([clean_text])
    combined_features = hstack([word_features, char_features])

    model = bundle["model"]
    prediction = str(model.predict(combined_features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(combined_features)[0]
        scores = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    else:
        scores = {prediction: 1.0}

    result = f"**Predicted category: {prediction}**"
    return scores, result


with gr.Blocks(title="Mind-Sorts | Text Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Mind-Sorts
        ### Mental-health-related text classification using NLP

        Enter a short text sample and the trained model will classify it into one of four
        predefined categories: **Anxiety, Depression, Normal, or Suicidal**.

        **How it works:** Text → Word + Character TF-IDF → Logistic Regression → Category
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to classify",
                placeholder="Type or paste a short text sample here...",
                lines=5,
            )
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an illustrative example")
            classify_button = gr.Button("Classify text", variant="primary")
            clear_button = gr.ClearButton(components=[text_input])
        with gr.Column(scale=2):
            prediction_output = gr.Markdown(label="Prediction")
            score_output = gr.Label(num_top_classes=4, label="Model scores by category")

    classify_button.click(
        fn=classify_text,
        inputs=text_input,
        outputs=[score_output, prediction_output],
    )
    text_input.submit(
        fn=classify_text,
        inputs=text_input,
        outputs=[score_output, prediction_output],
    )

    gr.Markdown(
        """
        ---
        **Important:** This is an educational text-classification demonstration, not a clinical
        assessment or diagnosis. The category and scores reflect the model's learned patterns
        and may be incorrect. Do not use this tool to make health or safety decisions.
        """
    )

if __name__ == "__main__":
    demo.launch()
