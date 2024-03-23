import cv2
import face_recognition
import os
import pickle
from tqdm import tqdm

images = []
classNames = []

ROOT = os.getcwd()
mylist = os.listdir("dataset")

for cl in mylist:
    c = os.path.join(os.path.join(ROOT, "dataset"), cl)
    for i in os.listdir(c):
        cur_img = cv2.imread(f'{c}/{i}')
        images.append(cur_img)
        classNames.append(cl)

def get_encodings(imgs):
    encodes = []
    for img in tqdm(imgs, desc="Encoding images"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img, num_jitters=2)
        if len(encoded_face) > 0:
            encoded_face = encoded_face[0]
            encodes.append(encoded_face)
    return encodes

encoded_face_train = get_encodings(images)

with open("encodings.pkl", "wb") as file:
    data = {"classNames": classNames, "encoded_face_train": encoded_face_train}
    pickle.dump(data, file)