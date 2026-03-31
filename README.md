# SmartFace Attendance System

A python-based facial recognition platform built for automated attendance logging. It connects a real-time web camera feed to a local database, identifying registered users as they walk by and recording their presence without requiring manual check-ins.

## 📝 Problem & Solution

**The Problem:** Traditional attendance tracking (like swiping ID cards or signing sheets) is slow, easily forgotten, and sometimes insecure (buddy punching). It also requires constant physical interaction.

**The Solution:** SmartFace removes the friction. By creating a lightweight dashboard that continuously monitors a video feed, users simply need to look at the camera. The system processes the image asynchronously, runs face detection and recognition, and logs the attendance directly to an SQLite database—all without stopping the video stream.

## ✨ Features

*   **Asynchronous Processing:** Video capturing runs on the main thread, while the heavy deep-learning model (`DeepFace`) processes frames in a background worker queue. This keeps the camera feed running smoothly at ~30 FPS without lagging the UI.
*   **Live Registration:** You can register new faces directly from the dashboard. It grabs the current frame buffer from the live feed rather than trying to reopen the busy webcam port.
*   **Multi-Camera Support:** Added a dropdown to switch between multiple connected cameras (e.g., laptop webcam vs. external USB camera or DroidCam) on the fly.
*   **False-Positive Filtering:** We tuned `minNeighbors` and `minSize` on the OpenCV face detector to prevent it from confusing background noise (like complex shirt patterns or posters) with actual faces.
*   **Debounced Logging:** The database layer has a built-in cooldown (e.g., 1 minute) so standing in front of the camera doesn’t flood the database with duplicate check-ins.

## ⚙️ How It Works (The Pipeline)

1. **Capture:** OpenCV pulls frames continuously.
2. **Detect:** We use Haar Cascades (tuned strictly) to quickly scan each frame for face bounding boxes.
3. **Queue:** If a face is found and the background worker is free, the frame is sent to the queue.
4. **Identify:** The background thread processes the face crop through DeepFace and checks for a biometric match against our SQLite database. 
5. **Log:** If matched within the confidence threshold (`0.40`), it inserts a timestamped record into the database and updates the UI.

## 🛠️ Project Structure

```text
smart-attendance-system/
├── config/
│   └── config.yaml          # Sensitivity thresholds and camera defaults
├── src/
│   ├── api/                 # Flask server & Dashboard 
│   │   ├── static/          # CSS styling and JS logic
│   │   ├── templates/       # HTML Pages
│   │   └── app.py           # Core routes
│   ├── core/
│   │   ├── face_recognition/ 
│   │   │   ├── face_detection.py # OpenCV scanner logic
│   │   │   └── identification.py # DeepFace model wrappers
│   │   └── pipeline.py           # Async threading system
│   └── database/
│       └── database.py           # SQLite handlers
└── README.md
```

## 🚀 Getting Started

### Prerequisites

You need Python 3.10 or 3.11 for compatibility with TensorFlow.

```bash
# Set up a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Force downgrade numpy to avoid specific array API crashes in TF/Pandas
pip install "numpy<2.0"
```

### Running the App

1. Make sure your Python path recognizes the `src` folder.
2. Run the application.

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = ".;$env:PYTHONPATH"; python src/api/app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 🚧 Challenges Faced 

*   **Blocking Main Thread:** Initially, running deep learning models on every frame caused the video stream to freeze terribly. Implementing a 1-item `queue.Queue` fixed this so the AI only takes the freshest frame while the video keeps moving.
*   **Camera Resource Conflicts:** Flask routes for "registration" originally tried to open the camera again while the stream was active. This crashed the app. We solved this by creating a globally shared `latest_frame` buffer.
*   **NumPy 2.x Breaks:** Many legacy ML packages (like TensorFlow 2.x) instantly broke under numpy 2.0. Locking it to `<2.0` was necessary to get DeepFace working.

## 🔮 Future Improvements

*   **Migrate Face Detection:** Swap out the legacy Haar Cascades for Google's MediaPipe BlazeFace for better robustness in different lighting conditions.
*   **Liveness Check:** Implement a blink-detector to prevent people from holding up photos in front of the lens.
*   **Production Deployment:** Wrap the Flask backend in `Gunicorn` or `Waitress` rather than using the default development server.
