import cv2
import numpy as np
import tensorflow as tf
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

detector = dlib.get_frontal_face_detector()
model = tf.keras.models.load_model('model.h5')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_dir = 'C:\\Users\\kazak\\Documents\\face\\face-rec\\Dataset\\'
target_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

def recognize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = img[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (224, 224))
        normalize = resized/255.0
        reshape = np.reshape(normalize, (1, 224, 224, 3))
        result = model.predict(reshape)
        label = train_generator.class_indices
        label = dict((v, k) for k, v in label.items())
        predicted = label[np.argmax(result)]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 47), 2)
        cv2.putText(img, str(predicted), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    recognize(frame)
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
