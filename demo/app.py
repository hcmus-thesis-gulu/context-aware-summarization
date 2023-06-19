import sys
import streamlit as st
from scripts.summarizer import VidSum

st.title('Video Summarization')

# Read hyperparameters from command line arguments
param1 = sys.argv[1]
param2 = sys.argv[2]

# Construct VidSum object with hyperparameters and store it in session state
if 'vidsum' not in st.session_state:
    st.session_state['vidsum'] = VidSum(param1, param2)

uploaded_file = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov'])
if uploaded_file is not None:
    st.video(uploaded_file)

    # Hyperparameters
    st.sidebar.header('Hyperparameters')
    param3 = st.sidebar.slider('Parameter 3', 0, 100, 50)
    param4 = st.sidebar.selectbox('Parameter 4', ['Option 1', 'Option 2', 'Option 3'])

    if st.button('Summarize!'):
        with st.spinner('Summarizing...'):
            summarized_video = st.session_state['vidsum'].summarize(uploaded_file, param3, param4)
        st.video(summarized_video)
