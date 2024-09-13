import streamlit as st
from PIL import Image
import numpy as np
from model import predict_emotion
from utils import crop_face, get_recommendations

# Set a title for the app
st.set_page_config(page_title="Emotion-Based Music Recommender", layout="wide")

# Display title and description
st.title("🎵 Emotion-Based Music Recommendation System")
st.write("Detect your emotion using your webcam and get personalized music recommendations based on how you feel!")

# Add sidebar for app control
st.sidebar.title("Control Panel")
st.sidebar.write("Use this control panel to start and stop the camera.")

# Add a button to start/stop the camera
run = st.sidebar.checkbox('Start Camera')

# Placeholder to display image frame from camera
FRAME_WINDOW = st.image([])

# Import necessary modules for webcam access
from io import BytesIO
import requests

# Dummy webcam URL for demonstration purposes
WEBCAM_URL = "http://your-webcam-url.com/latest.jpg"

# Function to simulate capturing image from webcam
def capture_frame():
    try:
        response = requests.get(WEBCAM_URL)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        st.error("Failed to access the webcam.")
        return None

# Define the structure of the page (columns for a clean UI)
col1, col2 = st.columns(2)

with col1:
    st.header("Your Live Feed")
    st.write("When the camera is running, you'll see your face here.")

with col2:
    st.header("Emotion Detection and Recommendations")
    st.write("Detected emotions will appear here, followed by song recommendations.")

emotion_list = []

if run:
    with col1:
        while run:
            # Capture frame from simulated webcam
            img = capture_frame()
            
            if img:
                img = img.convert('RGB')  # Ensure image is in RGB mode
                frame = np.array(img)  # Convert image to numpy array

                # Detect and crop face
                cropped_face = crop_face(frame)

                if cropped_face is not None:
                    # Predict emotion for the cropped face
                    emotion = predict_emotion(cropped_face)
                    emotion_list.append(emotion)

                # Display image feed in UI
                FRAME_WINDOW.image(img)

                if len(emotion_list) >= 40:
                    break
    
    with col2:
        # Display detected emotions
        st.write("Detected Emotions: ", emotion_list)
        
        # Sort emotions by frequency and remove duplicates
        unique_emotions = sorted(set(emotion_list), key=lambda x: emotion_list.count(x), reverse=True)
        st.write("Final Emotions: ", unique_emotions)
        
        # Get and display recommended songs
        recommendations = get_recommendations(unique_emotions)
        st.write("🎶 Recommended Songs:")
        for song in recommendations:
            st.write(f"- {song}")

else:
    st.write('Camera is turned off.')
