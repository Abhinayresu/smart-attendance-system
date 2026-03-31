import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from src.core.pipeline import AttendanceSystem
from src.utils.logger import logger
from src.database.database import DatabaseManager

"""
Flask Application for our Attendance Project.
This connects the web UI to our AI processing code.
"""

app = Flask(__name__)
# Create the system brain
smart_system = AttendanceSystem()
user_db = DatabaseManager()

# Controls
active_mode = True
camera_on = True
camera_id = 0
latest_frame = None

def stream_camera():
    """ Keeps grabbing frames from the webcam and processing them. """
    global active_mode, camera_on, camera_id, latest_frame
    webcam = None
    current_camera_id = camera_id
    retry_count = 0
    max_retries = 3
    
    try:
        while True:
            # 1. Handle Source Change
            if current_camera_id != camera_id:
                logger.info(f"Source change: Switching to camera ID {camera_id}")
                if webcam: webcam.release()
                webcam = None
                current_camera_id = camera_id
                retry_count = 0

            if not camera_on:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "SYSTEM STANDBY: CAMERA OFF", (120, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                cv2.waitKey(1000)
                continue

            if webcam is None or not webcam.isOpened():
                webcam = cv2.VideoCapture(current_camera_id, cv2.CAP_DSHOW)
                if not webcam.isOpened():
                    logger.warning(f"Could not open source {current_camera_id} (Attempt {retry_count+1}/{max_retries})")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, f"CAM {current_camera_id} UNAVAILABLE", (150, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        _, buffer = cv2.imencode('.jpg', error_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        cv2.waitKey(2000)
                        webcam = None
                        continue
                    
                    cv2.waitKey(1000)
                    continue

            ok, frame = webcam.read()
            if not ok:
                webcam.release()
                webcam = None
                continue
            
            latest_frame = frame.copy()

            if active_mode:
                try:
                    frame = smart_system.handle_video_frame(frame)
                except Exception as e:
                    logger.error(f"AI Stream Error: {str(e)}")
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        if webcam:
            webcam.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition', methods=['POST'])
def toggle():
    global active_mode
    settings = request.get_json()
    active_mode = settings.get('active', True)
    return jsonify({"status": "AI Active" if active_mode else "AI Paused"})

@app.route('/toggle_camera', methods=['POST'])
def toggle_cam():
    global camera_on
    settings = request.get_json()
    camera_on = settings.get('active', True)
    return jsonify({"status": "Camera On" if camera_on else "Camera Off"})

@app.route('/change_source', methods=['POST'])
def change_source():
    global camera_id
    settings = request.get_json()
    camera_id = int(settings.get('id', 0))
    return jsonify({"status": f"Source set to {camera_id}"})

@app.route('/register', methods=['POST'])
def enroll_member():
    """ Grabs the current frame from the persistent video stream and saves the face. """
    global latest_frame
    new_name = request.form.get('name')
    if not new_name:
        return jsonify({"error": "Name required"}), 400

    if latest_frame is None:
        return jsonify({"error": "Camera feed not ready"}), 500

    # Share the latest frame from the live feed
    frame = latest_frame.copy()

    # Step 1: Locate the face
    found_faces = smart_system.locator.find_face_boxes(frame)
    if len(found_faces) == 0:
        return jsonify({"error": "No face detected. Please look at the camera."}), 400
    
    # Step 2: Extract signature
    x, y, w, h = found_faces[0]
    face_crop = frame[y:y+h, x:x+w]
    fingerprint = smart_system.identifier.extract_face_fingerprint(face_crop)
    
    if fingerprint is None:
        return jsonify({"error": "Failed to create biometric eye-print. Try again."}), 500

    # Step 3: Save to DB
    if user_db.add_user(new_name, fingerprint):
        smart_system.load_registered_users() # Instant refresh
        return jsonify({"message": f"Successfully enrolled {new_name}!"})
    
    return jsonify({"error": "Database write failed."}), 500

@app.route('/logs')
def view_history():
    """ Fetches recent attendance to show on the dashboard. """
    try:
        import sqlite3
        with sqlite3.connect(user_db.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT u.name, a.timestamp 
                FROM attendance a 
                JOIN users u ON a.user_id = u.id 
                ORDER BY a.timestamp DESC 
                LIMIT 40
            ''')
            history = [{"name": r[0], "time": r[1]} for r in cur.fetchall()]
        return jsonify(history)
    except:
        return jsonify([])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
