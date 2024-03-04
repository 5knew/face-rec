import cv2
import os
import dlib

cap = cv2.VideoCapture(0)
count = 0
nameD = str(input("Name :")).lower()
path = 'C:\\Users\\kazak\\Documents\\face\\face-rec\\Dataset\\' + nameD

detector = dlib.get_frontal_face_detector()

isExists = os.path.exists(path)
if isExists:
    print("Already taken")
else:
    os.makedirs(path)

while True:
    ret, frame = cap.read()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgbFrame)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        count += 1
        name = path + '\\' + str(count) + '.jpg'
        if 0 <= y < rgbFrame.shape[0] and 0 <= y + h < rgbFrame.shape[0] and 0 <= rgbFrame.shape[1] and 0 <= x + w < \
                rgbFrame.shape[1]:
            print(f"Created Image: {count}.jpg ")
            cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,250), 2)
    cv2.imshow("DataSet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count > 159:
        break
cap.release()
cv2.destroyAllWindows()
