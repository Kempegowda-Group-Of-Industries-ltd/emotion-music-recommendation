import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained emotion detection model (change path if needed)
model = tf.keras.models.load_model('path_to_your_trained_model.h5')

# List of emotions (replace with the actual labels used by your model)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_img):
    # Convert the PIL image to a NumPy array
    face_img = np.array(face_img)
    
    # Convert the image to grayscale
    if len(face_img.shape) == 3:  # Check if image has 3 channels (RGB)
        face_img = np.mean(face_img, axis=-1)  # Convert to grayscale
    
    # Resize the image to 48x48 pixels
    face_img = Image.fromarray(face_img).resize((48, 48))
    
    # Convert image to NumPy array and normalize
    face_img = np.array(face_img).astype('float32') / 255
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension for grayscale image

    # Predict emotion
    predictions = model.predict(face_img)
    max_index = np.argmax(predictions[0])
    emotion = EMOTIONS[max_index]
    
    return emotion
