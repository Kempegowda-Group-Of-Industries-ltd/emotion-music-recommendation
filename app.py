import streamlit as st
import cv2
import numpy as np
from model import predict_emotion
from utils import crop_face, get_recommendations

# Streamlit app setup
st.title("Emotion-Based Music Recommendation System")

# Initialize the camera feed
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

if run:
    emotion_list = []

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and crop the face
        cropped_face = crop_face(frame)
        
        if cropped_face is not None:
            # Predict emotion for the cropped face
            emotion = predict_emotion(cropped_face)
            emotion_list.append(emotion)
        
        FRAME_WINDOW.image(frame)

        if len(emotion_list) >= 40:  # Process for 40 frames
            break

    # Post-processing of emotions
    st.write("Detected Emotions: ", emotion_list)
    
    # Sort emotions by frequency and remove duplicates
    unique_emotions = sorted(set(emotion_list), key=lambda x: emotion_list.count(x), reverse=True)
    st.write("Final Emotions: ", unique_emotions)

    # Recommend songs based on emotions
    recommendations = get_recommendations(unique_emotions)
    st.write("Recommended Songs: ", recommendations)
else:
    st.write('Stopped')






import streamlit as st
import cv2
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

# Placeholder to display video frame from camera
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

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
            # Capture frame from camera
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect and crop face
            cropped_face = crop_face(frame)
            
            if cropped_face is not None:
                # Predict emotion from cropped face
                emotion = predict_emotion(cropped_face)
                emotion_list.append(emotion)

            # Display video feed in UI
            FRAME_WINDOW.image(frame)

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

# Release the camera when stopping
if not run:
    camera.release()
