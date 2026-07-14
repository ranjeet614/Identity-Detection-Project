FACENET

!pip install mtcnn
!pip install keras-facenet

from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import os

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

!wget -O virat.jpg "https://i.imgur.com/EXN9G9O.jpg"

print("✔ Image downloaded!")

detector = MTCNN()
embedder = FaceNet()

def extract_face(img_path, size=(160,160)):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Cannot load image file.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        print("❌ No face detected.")
        return None

    x, y, w, h = faces[0]['box']
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    return face

face = extract_face("virat.jpg")

if face is not None:
    face_expanded = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face_expanded)[0]

    print("\n✔ FaceNet embedding for Virat Kohli:")
    print(embedding)

    print("\n✔ Displaying image:")
    cv2_imshow(cv2.imread("virat.jpg"))
else:
    print("❌ Face extraction failed.")
