from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv() 


class TranscriptNotAvailableError(Exception):
    """Raised when a video has no usable transcript."""
    pass


def extract_video_id(video_link: str) -> str:
    """Parses the actual URL structure instead of guessing from substrings —
    handles /watch, youtu.be, /shorts/, /embed/, /live/ correctly, including
    extra query params like &t=30s."""
    parsed = urlparse(video_link)

    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")

    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            if "v" in query:
                return query["v"][0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed.path.split("/")[2]

    raise ValueError(f"Could not extract a video ID from: {video_link}")


def preprocess(user_link):
    """Now returns (fetch_text, video_id) instead of just fetch_text —
    video_id is needed downstream to scope the vector store per video."""
    fetch_text = []

    video_id = extract_video_id(user_link)

    y_t_api = YouTubeTranscriptApi()

    try:
        fetch_trans = y_t_api.fetch(video_id, languages=['en'])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        raise TranscriptNotAvailableError(
            f"Subtitles not available for video: {video_id}"
        )

    for i in fetch_trans:
        fetch_text.append(i.text)

    return fetch_text, video_id


def text_chunk(fetch_text):
    print("Chunking text...")

    full_transcript = " ".join(fetch_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    documents = [Document(page_content=full_transcript)]
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def local_db(chunks, video_id: str):
    """Now requires video_id — scopes each video into its own Chroma
    collection so questions about one video can't retrieve chunks from
    another (this was the cross-video contamination bug)."""
    print(f"Creating local vector database for video {video_id}...")

    local_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    collection_name = f"video_{video_id}"

    vector_db_local = Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        collection_name=collection_name,
        persist_directory="/app/backend/chroma_db_local"
    )

    print(f"Local vector database created for video {video_id}.")

    return vector_db_local


def instruct(vector_db_local):
    retriever = vector_db_local.as_retriever(search_kwargs={"k": 2})

    template = """
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    return retriever, prompt


def generate(retriever, prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


def user_input(user_question, rag_chain):
    response = rag_chain.invoke(user_question)

    if hasattr(response, "content"):
        return response.content

    return str(response)
