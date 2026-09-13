from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings


def preprocess():
    video_link = input('enter the video link : ')

    video_id = ''
    if 'list' in video_link:
        video_id+= video_link.split('=')[1].split('&')[0]
    elif '.be/' in video_link:
        video_id +=video_link.split('.be/')[1].split('?')[0]
    else:
        video_id+= video_link.split('=')[1]


    y_t_api = YouTubeTranscriptApi()

    try: 
        fetch_trans = y_t_api.fetch(video_id, languages=['en'])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        print('Subtitles not available ...')
        sys.exit()



    fetch_text = []
    for i in fetch_trans:
        fetch_text.append(i.text)

            # to see subtitles at time 
    # print(fetch_trans.to_raw_data())



    os.environ["GOOGLE_API_KEY"] = "Api_key_here"       # setting the gemini api key, if you are in colab 
    # if in local , then set it in environment variables 

    return fetch_text



def text_chunk():

    fetch_text = preprocess() 

    print("Chunking text...")

    full_transcript = " ".join(fetch_text)    # Join all transcript lines into one large string to allow proper chunking

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    documents = [Document(page_content=full_transcript)] 
    chunks = text_splitter.split_documents(documents)       # Spliting the  document into chunks

    print(f"Created {len(chunks)} chunks.")

    return chunks






def local_db():

    chunks = text_chunk()

    print("Creating local vector database using SentenceTransformer embeddings...")

    local_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")     # select huggingface embedding model 

    # Recreate the vector database with the local embeddings
    vector_db_local = Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        persist_directory="./chroma_db_local" # directory path 
    )
    print("local Vector database created successfully using SentenceTransformer.")

    return vector_db_local






def instruct():

    vector_db_local = local_db()

    # Configure the database to act as a document retriever
    retriever = vector_db_local.as_retriever(search_kwargs={"k": 2})    # k , is for getting simila chunks     

    # Defineing the hidden prompt structure for the LLM
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







def generate():

    retriever , prompt = instruct()

        #  model needs to e change according to api_key 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Helper function to stitch retrieved chunks into a single text block
    def format_docs(docs):      # formating chunk into text , so model can read it 
        return "\n\n".join(doc.page_content for doc in docs)

    # Connect everything together using LangChain Expression Language (LCEL)
    rag_chain = (           
        {"context": retriever | format_docs, 
        "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain




def user_input():
    rag_chain = generate()


    print("\n--- PDF Chatbot Initialized ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_question = input("\nYour Question: ")

        if user_question.lower() in ['exit', 'quit']:       # to break the loop 
            print("\n Shutting down chatbot....")
            break

        response = rag_chain.invoke(user_question)          # sending question 

        if isinstance(response.content, list):            # setting output format 
            clean_answer = response.content[0]['text']
        else:
            clean_answer = response.content

        print(f"Answer: {clean_answer}")


user_input()