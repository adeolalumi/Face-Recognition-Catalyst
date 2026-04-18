# Face-Catalyst
Face Recognition System
Key Features:

Real-Time Detection: Uses a threaded camera loop to identify faces instantly.
Performance Optimized: Implements 4x downscaling on frames before processing to ensure high FPS even on modest hardware.
Local SQLite Database: Stores user profiles and image paths locally for persistent recognition across sessions.
Instant Enrollment: Features a built-in UI popup to capture new faces, crop them automatically, and save them to the known database without restarting.
Distance-Based Accuracy: Uses Euclidean distance (threshold: 0.48) to minimize false positives.

This is typical for a lightweight, offline face recognition app built with OpenCV + face_recognition library + SQLite.
