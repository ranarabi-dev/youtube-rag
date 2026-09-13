import streamlit as st
import requests

st.set_page_config(page_title="Youtube Video Q&A",  layout="wide")


st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Global Styles */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #ff0000, #c25e5e, #281713, #530004);
        background-size: 400% 400%;
        animation: gradient 8s ease infinite;
        color: white;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Container Effect */
    .css-1d391kg { /* Main container padding */
        padding-top: 2rem;
    }

    /* Style the Input Container */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction"] {
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 5px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
    }

    /* Input Fields Styling */
    label {
        font-weight: 600;
        color: #fff !important;
    }
    
    /* Dropdown & Number Inputs background */
    div[data-baseweb="select"] span, 
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 5em;
        font-weight: bold;
        transition: all 0.9s ease;
        background-color: #ffffff;
        color: #23a6d5;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background-color: #b3b3b3;
        font-color:white
    }


    /* Success/Error Message Styling */
    .stSuccess, .stError {
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)



st.title("Youtube Video Q&A")


def reset_chat():
    st.session_state.messages=[]

# Session State
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []



# link 
with st.container(border=True):
    st.subheader("Load YouTube Video")

    video_link = st.text_input("Enter YouTube Video Link",  key="video_link_input")

    if st.button("Send Link", use_container_width=True):
        reset_chat()
        if not video_link:
            st.warning("Please enter a YouTube link.")
            st.stop()

        try:
            # response = requests.post("http://127.0.0.1:8000/video_link_from", json={"video_link": video_link})
            response = requests.post("http://backend:8000/video_link_from", json={"video_link": video_link})

            if response.status_code == 200:
                st.success("Video processed successfully.")
                st.session_state.video_loaded = True

            else:
                st.error(f"HTTP Error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")





# Chat Section
if st.session_state.video_loaded:

    st.markdown("---")
    st.subheader("Ask Questions About The Video")

    # Show previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    user_question = st.chat_input("Ask something about the video...")

    if user_question:
        # Save user message
        st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching answer..."):

                try:
                    # response = requests.post( "http://127.0.0.1:8000/predict", json={"question": user_question })
                    response = requests.post( "http://backend:8000/predict", json={"question": user_question })

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get( "Prediction", "No answer found.")
                        st.write(answer)

                        st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })

                    else:
                        st.error(f"HTTP Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {e}")