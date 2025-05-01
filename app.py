from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load known faces
path = 'known_faces'
images = []
names = []
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path, filename))
    images.append(img)
    names.append(os.path.splitext(filename)[0])

# Encode faces
def find_encodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

known_encodings = find_encodings(images)

# Attendance dictionary
attendance_dict = {}

def mark_attendance(name):
    if name not in attendance_dict:
        now = datetime.now()
        dt_string = now.strftime('%H:%M:%S')
        attendance_dict[name] = dt_string

        # Save attendance immediately
        if not os.path.exists('Attendance_Files'):
            os.makedirs('Attendance_Files')

        today_date = datetime.now().strftime('%Y-%m-%d')
        file_name = f"Attendance_{today_date}.xlsx"
        file_path = os.path.join('Attendance_Files', file_name)

        # Update the Excel file
        attendance_df = pd.DataFrame(list(attendance_dict.items()), columns=['Name', 'Time'])
        attendance_df.to_excel(file_path, index=False)
        print(f"Attendance saved: {name} at {dt_string}")

# Open webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img_small = cv2.resize(img, (0,0), None, 0.25, 0.25)
            img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            faces_current_frame = face_recognition.face_locations(img_small)
            encodings_current_frame = face_recognition.face_encodings(img_small, faces_current_frame)

            for encodeFace, faceLoc in zip(encodings_current_frame, faces_current_frame):
                matches = face_recognition.compare_faces(known_encodings, encodeFace)
                face_distances = face_recognition.face_distance(known_encodings, encodeFace)
                match_index = np.argmin(face_distances)

                if matches[match_index]:
                    name = names[match_index].upper()
                    mark_attendance(name)

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     # Define the faculty_name and current_date (you can replace this with dynamic data from your database or context)
#     faculty_name = "John Doe"  # Example faculty name, replace with your dynamic data
#     current_date = datetime.now().strftime('%Y-%m-%d')  # Current date formatted as yyyy-mm-dd
#     return render_template('attendance.html', faculty_name=faculty_name, current_date=current_date)

@app.route('/')
def home():
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/attendance')
def attendance():
    faculty_name = "John Doe"  # Or fetch from session/context
    current_date = datetime.now().strftime('%Y-%m-%d')
    class_name = "Class 1"     # Optional: dynamic from query
    return render_template('attendance.html', faculty_name=faculty_name, current_date=current_date, class_name=class_name)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save-attendance', methods=['POST'])
def save_attendance():
    if not os.path.exists('Attendance_Files'):
        os.makedirs('Attendance_Files')

    today_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"Attendance_{today_date}.xlsx"
    file_path = os.path.join('Attendance_Files', file_name)

    attendance_df = pd.DataFrame(list(attendance_dict.items()), columns=['Name', 'Time'])
    attendance_df.to_excel(file_path, index=False)

    return jsonify({'message': 'Attendance saved successfully!', 'file_path': file_path})

if __name__ == "__main__":
    app.run(debug=True)
