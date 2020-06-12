import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'Attendence Project'

images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cls in mylist:
    cur_Img = cv2.imread(f'{path}/{cls}')
    images.append(cur_Img)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img1)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attenence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeList_Known = findEncodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgs1 = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    faces_current_frame = face_recognition.face_locations(imgs1)
    encode = face_recognition.face_encodings(imgs1, faces_current_frame)

    for encode_face, face_loc in zip(encode, faces_current_frame):
        matches = face_recognition.compare_faces(encodeList_Known, encode_face)
        face_dis = face_recognition.face_distance(encodeList_Known, encode_face)
        print(face_dis)
        matchindex = np.argmin(face_dis)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            markAttendance(name)
    cv2.imshow('Webcome', img)
    cv2.waitKey(1)


