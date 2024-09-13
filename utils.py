from PIL import Image
import face_recognition

# Function to crop face using face_recognition library
def crop_face(frame):
    # Convert the image to RGB format
    rgb_frame = frame.convert('RGB')
    
    # Convert the PIL image to a NumPy array
    frame_array = np.array(rgb_frame)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(frame_array)
    
    if len(face_locations) == 0:
        return None

    # Use the first detected face
    top, right, bottom, left = face_locations[0]
    cropped_face = frame.crop((left, top, right, bottom))
    return cropped_face

# Recommend songs based on detected emotions
def get_recommendations(emotion_list):
    song_dict = {
        'Happy': ['Song 1', 'Song 2', 'Song 3'],
        'Sad': ['Song 4', 'Song 5', 'Song 6'],
        'Angry': ['Song 7', 'Song 8', 'Song 9'],
        'Surprise': ['Song 10', 'Song 11', 'Song 12'],
        'Neutral': ['Song 13', 'Song 14', 'Song 15'],
        'Fear': ['Song 16', 'Song 17', 'Song 18'],
        'Disgust': ['Song 19', 'Song 20', 'Song 21']
    }

    recommendations = []
    for emotion in emotion_list:
        recommendations += song_dict.get(emotion, ['Default Song'])

    return recommendations
