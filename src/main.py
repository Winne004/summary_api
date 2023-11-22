import logging
from typing import Annotated, List
from transformers import pipeline
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse


def initialise_ner_ai_model():
    return pipeline("ner", grouped_entities=True)


def initialise_zero_shot_ai_model():
    return pipeline("zero-shot-classification")


logging.warning("Downloading model, this could take some time....")
ner = initialise_ner_ai_model()
zero_shot_model = initialise_zero_shot_ai_model()
logging.warning("Finished Downloading model. App should now be good to go.")

app = FastAPI()


class TextToExtractEntities(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_entities/")
def create_item(text_to_extract_entities: TextToExtractEntities):
    named_entities = ner(text_to_extract_entities.text)
    for entity in named_entities:
        entity["score"] = float(entity["score"])
    return JSONResponse(named_entities)


@app.get("/get_classification/")
def read_item(
    text_to_classify: str, classification_labels: Annotated[List[str], Query()] = None
):
    labels = zero_shot_model(text_to_classify, classification_labels)

    return labels
