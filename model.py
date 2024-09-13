import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained emotion detection model (change path if needed)
model = tf.keras.models.load_model('path_to_your_trained_model.h5')

# List of emotions (replace with the actual labels used by your model)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_img):
    # Preprocess the image to the input format of the model
    face_img = cv2.resize(face_img, (48, 48))  # Resize to 48x48
    face_img = face_img.astype('float32') / 255
    face_img = np.expand_dims(face_img, axis=0)

    # Predict emotion
    predictions = model.predict(face_img)
    max_index = np.argmax(predictions[0])
    emotion = EMOTIONS[max_index]
    
    return emotion
