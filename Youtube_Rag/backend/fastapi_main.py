from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

vector_db, rag_chain = None, None

class VideoRequest(BaseModel):
    video_link: str

class QuestionRequest(BaseModel):
    question: str


@app.post("/video_link_from")
def video_link_from(data: VideoRequest):
    global vector_db, rag_chain

    import backend.Youtube_Q_A_bot

    transcript = backend.Youtube_Q_A_bot.preprocess(data.video_link)
    chunks = backend.Youtube_Q_A_bot.text_chunk(transcript)
    vector_db = backend.Youtube_Q_A_bot.local_db(chunks)
    retriever, prompt = backend.Youtube_Q_A_bot.instruct(vector_db)
    rag_chain = backend.Youtube_Q_A_bot.generate(retriever, prompt)

    return {"message": "Video processed successfully"}


@app.post("/predict")
def predict(data: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        return {"error": "Video not loaded yet"}

    response = rag_chain.invoke(data.question)

    if hasattr(response, "content"):
        return {"Prediction": response.content}

    return {"Prediction": str(response)}