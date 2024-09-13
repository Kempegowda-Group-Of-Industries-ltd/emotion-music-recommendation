import cv2

# Function to crop face using OpenCV's Haarcascade model
def crop_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    cropped_face = frame[y:y + h, x:x + w]
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

