from fastapi import FastAPI
import gradio as gr

import pandas as pd

from transformers import pipeline
from gradio.components import Textbox

# Load the sentiment analysis pipeline with DistilBERT
distilbert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
label_map = {"POSITIVE":"other", "NEGATIVE":"sensitive"}

input1 = Textbox(lines=2, placeholder="Type your text here...")
file_upload = gr.File(file_types=[".xlsx"])

CUSTOM_PATH = "/talktoloop"

app = FastAPI()


@app.get("/")
def read_main():
    return {"message": "This is your main app"}

def predict_sentiment(file_upload):
        """
        Predicts the sentiment of the input text using DistilBERT.
        :param text: str, input text to analyze.
        :return: str, predicted sentiment and confidence score.
        """
        input_df = pd.read_excel(file_upload)
        all_data = list()
        for idx, row in input_df.iterrows():
              
              dict_key = dict()
              
              result = distilbert_pipeline(row["text"])[0]

              #dict_key["Sr. No."] = row["Sr. No."]
              dict_key["text"] = row["text"]
              dict_key["actual_label"] = row["actual_label"]
              dict_key["label"] = label_map[result['label']]
              dict_key["conf"] = result['score']

              all_data.append(dict_key)

        filepath = "./output_file.xlsx"
        pd.DataFrame(all_data).to_excel(filepath, index=False)
        return filepath
        # result = distilbert_pipeline(text)[0]
        # label = label_map[result['label']]
        # score = result['score']
        # return f"TAG: {label}, Confidence: {score:.2f}"

# Create a Gradio interface
text_input = gr.Interface(fn=predict_sentiment,
                     inputs=file_upload,
                     outputs= gr.File(label="Inference Output"),
                     title="Talk2Loop Sensitive statement tags",
                     description="This model predicts the sensitivity of the input text. Enter a sentence to see if it's sensitive or not.")

#io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, text_input, path=CUSTOM_PATH)
