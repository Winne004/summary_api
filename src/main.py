import logging
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


def initialise_ai_model():
    logging.warning("Downloading model, this could take some time....")
    summarizer = pipeline(model="google/pegasus-xsum")
    logging.warning("Finished Downloading model. App should now be good to go.")
    return summarizer


summarizer = initialise_ai_model()


class TextToSummarise(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/summarise/")
def create_item(text_to_summarise: TextToSummarise):
    summary = summarizer([text_to_summarise.text[:500]])
    summarized_text = summary[0]["summary_text"]

    return {"summarized_text": summarized_text}
