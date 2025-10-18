from transformers import pipeline
import gradio as gr
import pandas as pd

# 📘 Загружаем модель анализа тональности
analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 🧠 Функция анализа текста
def analyze_review(text):
    result = analyzer(text)
    label = result[0]["label"]
    score = round(result[0]["score"], 3)
    return f"Prediction: {label}\nConfidence: {score}"

# 💬 Интерфейс Gradio
demo = gr.Interface(
    fn=analyze_review,
    inputs=gr.Textbox(label="Enter a movie review"),
    outputs=gr.Textbox(label="AI Sentiment Result"),
    title="🎬 Movie Review Analyzer",
    description="Analyze the sentiment of movie reviews using a multilingual BERT model from Hugging Face."
)

if __name__ == "__main__":
    demo.launch()
