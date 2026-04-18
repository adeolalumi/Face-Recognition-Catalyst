
import cv2
import os
import sqlite3
import numpy as np
from datetime import datetime
import face_recognition  # pyright: ignore[reportMissingImports]

# Kivy UI Imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class FaceRecogApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.image_display = Image()
        self.layout.add_widget(self.image_display)

        self.capture_button = Button(
            text='ENROLL CURRENT NEW FACE',
            size_hint=(1, 0.15),
            font_size='20sp'
        )
        self.capture_button.bind(on_press=self.open_enrollment_popup)
        self.layout.add_widget(self.capture_button)

        self.save_path = "known_faces"
        os.makedirs(self.save_path, exist_ok=True)

        # Database - now in the same folder as your .py file
        self.setup_sqlite()

        self.capture = cv2.VideoCapture(0)
        self.last_frame = None

        self.known_encodings = []
        self.known_names = []

        self.load_faces_from_db()

        Clock.schedule_interval(self.update, 1.0 / 15.0)

        return self.layout

    # ---------------- DATABASE ----------------
    def setup_sqlite(self):
        # This gets the folder where FaceRecognitionCatalyst.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "faces.db")
        
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                image_path TEXT,
                created_at TEXT
            )
        ''')
        self.conn.commit()
        
        print(f"✅ Database connected at: {db_path}")

    def load_faces_from_db(self):
        self.known_encodings.clear()
        self.known_names.clear()

        self.cursor.execute("SELECT name, image_path FROM users")
        rows = self.cursor.fetchall()

        for name, path in rows:
            if os.path.exists(path):
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(name)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        print(f"Total faces loaded: {len(self.known_names)}")

    # ---------------- RECOGNITION ----------------
    def recognize_face(self, face_encoding):
        if not self.known_encodings:
            return "Unknown"

        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_index = np.argmin(distances)

        if distances[best_index] < 0.48:
            return self.known_names[best_index]
        return "Unknown"

    # ---------------- CAMERA LOOP (Much Faster) ----------------
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.last_frame = frame.copy()

        # Downscale for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        scale = 4
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            name = self.recognize_face(encoding)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_display.texture = texture

    # ---------------- ENROLLMENT ----------------
    def open_enrollment_popup(self, instance):
        if self.last_frame is None:
            print("Waiting for camera...")
            return

        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text="Enter Name:"))
        self.name_input = TextInput(multiline=False)
        content.add_widget(self.name_input)

        btn = Button(text="SAVE FACE")
        btn.bind(on_release=self.process_enrollment)
        content.add_widget(btn)

        self.popup = Popup(title="Enroll User", content=content, size_hint=(0.8, 0.4))
        self.popup.open()

    def process_enrollment(self, instance):
        name = self.name_input.text.strip()
        if not name or self.last_frame is None:
            return

        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)

        if not face_locations:
            print("No face detected. Try better lighting or angle.")
            return

        top, right, bottom, left = face_locations[0]
        margin = 40
        h, w = self.last_frame.shape[:2]
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(h, bottom + margin)
        right = min(w, right + margin)

        face_crop = self.last_frame[top:bottom, left:right]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        path = os.path.join(self.save_path, filename)

        cv2.imwrite(path, face_crop)

        self.cursor.execute(
            "INSERT INTO users (name, image_path, created_at) VALUES (?, ?, ?)",
            (name, path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        self.conn.commit()

        print(f"Saved face for: {name}")
        self.load_faces_from_db()
        self.popup.dismiss()

    def on_stop(self):
        self.capture.release()
        self.conn.close()


if __name__ == "__main__":
    FaceRecogApp().run()