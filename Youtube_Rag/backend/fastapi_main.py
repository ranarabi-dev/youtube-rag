from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import backend.Youtube_Q_A_bot as yt_bot
from backend.Youtube_Q_A_bot import TranscriptNotAvailableError

app = FastAPI()

vector_db, rag_chain = None, None

class VideoRequest(BaseModel):
    video_link: str

class QuestionRequest(BaseModel):
    question: str


@app.post("/video_link_from")
def video_link_from(data: VideoRequest):
    global vector_db, rag_chain

    try:
        transcript, video_id = yt_bot.preprocess(data.video_link)
    except TranscriptNotAvailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = yt_bot.text_chunk(transcript)
    vector_db = yt_bot.local_db(chunks, video_id)
    retriever, prompt = yt_bot.instruct(vector_db)
    rag_chain = yt_bot.generate(retriever, prompt)

    return {"message": "Video processed successfully"}


@app.post("/predict")
def predict(data: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Video not loaded yet")

    response = rag_chain.invoke(data.question)

    if hasattr(response, "content"):
        return {"Prediction": response.content}

    return {"Prediction": str(response)}
