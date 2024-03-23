import cv2
import face_recognition
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import datetime

encoded_face_train = []
classNames = []

with open("encodings.pkl", "rb") as encodings:
    p = pickle.load(encodings)
    encoded_face_train = p["encoded_face_train"]
    classNames = p["classNames"]

def detect(img, upsample=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS, number_of_times_to_upsample=upsample, model="cnn")
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    present = []
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(50, 168, 82),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (50, 168, 82), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            if name not in present:
                present.append(name)
    return (present, img)

def main():
    st.title("Attendance System")
    upload_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if upload_image is not None:
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        upsample = st.slider('Not detecting enough? Upsample image:', 1, 5, 1)

        res, temp = detect(image, upsample)
        st.image(temp, channels="RGB", caption="Present")

        st.write("Present as of ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sno = pd.Index(list(range(1, len(res) + 1)))
        df = pd.DataFrame({"Name": res})
        df = df.set_index([sno])
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()