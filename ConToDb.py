import face_recognition
import numpy as np
import psycopg2
import os
import cv2

# Подключение к базе данных PostgreSQL
conn = psycopg2.connect(dbname="face_recognition", user="postgres", password="123", host="localhost", port="5432")
cur = conn.cursor()

# Создание таблиц, если они не существуют
cur.execute("""
    CREATE TABLE IF NOT EXISTS individuals (
        id SERIAL PRIMARY KEY,
        name VARCHAR
    )
""")
cur.execute("""
    CREATE TABLE IF NOT EXISTS encodings (
        id SERIAL PRIMARY KEY,
        individual_id INTEGER REFERENCES individuals(id),
        encoding BYTEA
    )
""")
conn.commit()

def add_individual_with_encoding(name, encoding):
    # Вставка нового лица
    cur.execute("INSERT INTO individuals (name) VALUES (%s) RETURNING id", (name,))
    individual_id = cur.fetchone()[0]
    conn.commit()

    # Конвертация кодировки в формат, подходящий для PostgreSQL
    encoding_bytes = psycopg2.Binary(np.array(encoding).tobytes())

    # Вставка кодировки, связанной с человеком
    cur.execute("INSERT INTO encodings (individual_id, encoding) VALUES (%s, %s)", (individual_id, encoding_bytes))
    conn.commit()

def load_and_train(directory="known_faces"):
    for name in os.listdir(directory):
        person_dir = os.path.join(directory, name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                add_individual_with_encoding(name, encodings[0])

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        cur.execute("SELECT name, encoding FROM individuals JOIN encodings ON individuals.id = encodings.individual_id")
        for row in cur.fetchall():
            name, encoding_bytes = row
            encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

    def recognize_faces(self, image_to_check):
        unknown_image = face_recognition.load_image_file(image_to_check)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        names = []
        for unknown_encoding in unknown_encodings:
            results = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding)
            name = "Unknown"
            if True in results:
                first_match_index = results.index(True)
                name = self.known_face_names[first_match_index]
            names.append(name)
        return names

def recognize_faces_in_frame(frame, recognizer):
    # Преобразование цветового пространства из BGR (OpenCV) в RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Находим все лица на кадре
    face_locs = face_recognition.face_locations(rgb_frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

    face_names = []
    for face_encoding in face_encs:
        # Используйте ваш класс FaceRecognizer для распознавания лиц
        names = recognizer.recognize_faces(face_encoding)  # Убедитесь, что этот метод работает с одной кодировкой лица
        face_names.extend(names)

    return face_locs, face_names

# Главный цикл для обработки видеопотока
if __name__ == '__main__':
    recognizer = FaceRecognizer()  # Создание экземпляра FaceRecognizer
    video_capture = cv2.VideoCapture(0)  # Захват видео с первой подключенной камеры

    while True:
        ret, frame = video_capture.read()
        face_locations, face_names = recognize_faces_in_frame(frame, recognizer)

        # Отображение результатов на кадре
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
