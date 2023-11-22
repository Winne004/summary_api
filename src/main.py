import logging
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

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
