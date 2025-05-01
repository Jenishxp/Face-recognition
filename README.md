# 👤 Face Recognition Attendance System

This is a real-time face recognition-based attendance system built using Python, OpenCV, Flask, and face_recognition. The application detects and recognizes faces from a webcam feed, and automatically logs attendance to an Excel sheet.

---

## 📸 Features

- Real-time face recognition via webcam
- Automatically logs attendance when a face is recognized
- Saves daily attendance in an Excel file
- Web interface built using Flask
- Face data managed from the `known_faces` directory

---

## 🛠️ Tech Stack

- **Frontend**: HTML (Jinja2 Templates)
- **Backend**: Flask (Python)
- **Libraries**: OpenCV, face_recognition, NumPy, Pandas

---

## 📂 Project Structure

Face-recognition/ 
├── app.py # Main Flask application 
├── templates/ 
│   ├── login.html 
│   ├── dashboard.html 
│   └── attendance.html 
├── known_faces/ # Folder with known face images 
├── Attendance_Files/ # Output Excel files (auto-created) 
└── requirements.txt # List of required Python packages

## 🚀 How to Run

### 1. Clone the Repo
project:
  name: Face Recognition Attendance System
  author: Jenish Patel
  license: MIT
  description: >
    A real-time face recognition-based attendance system using Flask, OpenCV, and face_recognition.
    Automatically logs attendance to Excel when a face is detected through the webcam.

requirements:
  python_version: ">=3.7"
  dependencies_file: requirements.txt
  special_note: You may need to install dlib separately depending on your system.

setup:
  steps:
    - step: Clone the Repository
      command: git clone https://github.com/Jenishxp/Face-recognition.git
    - step: Navigate to Project Folder
      command: cd Face-recognition
    - step: Install Dependencies
      command: pip install -r requirements.txt
    - step: Add Known Faces
      description: >
        Put clear face images (JPG/PNG) in the `known_faces` folder.
        The filename (without extension) will be used as the person's name.
    - step: Run the App
      command: python app.py
    - step: Open in Browser
      url: http://127.0.0.1:5000/

output:
  folder: Attendance_Files
  format: Attendance_YYYY-MM-DD.xlsx
  description: >
    Every time a new face is detected, attendance is logged into an Excel file inside the Attendance_Files directory.

future_improvements:
  - Add user login and authentication
  - Admin dashboard to view or export attendance history
  - Cloud integration for saving attendance online