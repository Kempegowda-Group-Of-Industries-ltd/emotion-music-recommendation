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
