# Face Recognition Catalyst

Lightweight offline face recognition app for real-time webcam recognition and fast local enrollment.

This project demonstrates a practical computer vision workflow: capture video frames, detect and encode faces, compare embeddings, store user profile data locally, and keep the interface responsive during recognition.

## Highlights

- Real-time webcam recognition with a threaded camera loop
- Frame downscaling before recognition to improve FPS on modest hardware
- Instant enrollment workflow for adding new faces without restarting the app
- Local SQLite storage for user profiles and image paths
- Euclidean distance matching with a configurable threshold to reduce false positives

## Tech Stack

Python, OpenCV, face_recognition, SQLite

## Project Structure

```text
Face_Catalyst.py       Main application logic
faces.db               Local SQLite database used by the app
.github/               GitHub metadata and workflows
```

## How To Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install opencv-python face-recognition numpy
python Face_Catalyst.py
```

`face-recognition` depends on `dlib`, which may require Visual Studio Build Tools on Windows if a compatible wheel is not available.

## What This Shows

- Real-time computer vision pipeline design
- Local-first face recognition without a cloud dependency
- Enrollment and recognition workflows in one app
- Practical performance tuning for camera-based applications

## Next Improvements

- Add a locked `requirements.txt`
- Move generated databases and face assets out of source control
- Add screenshots or a short demo GIF
- Add configuration for camera index, threshold, and database path