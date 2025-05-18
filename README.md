# AI Attendance System

An AI-powered facial recognition-based attendance management system built using **Flask**, **InsightFace**, **OpenCV**, and **MySQL**. This system enables administrators to manage, verify, and export attendance data with real-time face recognition technology.

---

## 💻 Tech Stack

### 🧠 Machine Learning
- **InsightFace**: Face detection & recognition
- **OpenCV**: Real-time image processing
- **NumPy & Pandas**: Data handling & processing

### 🧰 Backend
- **Python (Flask)**: Web framework
- **MySQL**: Relational database
- **Werkzeug**: Password hashing

### 🌐 Frontend
- HTML, CSS, JavaScript (for basic UI and form handling)

---

## ✅ Features

- Real-time facial recognition for marking attendance
- Admin authentication & login system
- Admin dashboard to:
  - View attendance records
  - Export attendance to Excel
  - Manage and approve new student registrations
- Multiple image training using InsightFace embeddings
- Database integration with MySQL for persistent data storage

---

## 📁 Project Structure

AI-Attendance-system/
│
├── attendance_records/ # Excel files for exported attendance
├── face_data/ # Face embedding files for trained faces
├── static/ # Static assets (CSS, JS)
├── templates/ # HTML templates
├── app.py # Main Flask application
├── connection.py # MySQL connection setup
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Tarun-428/AI-Attendance-system.git
cd AI-Attendance-system
2. Create a Virtual Environment (Optional)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate       # For Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Configure the MySQL Database
Create a database (e.g., attendance_system)

Edit connection.py with your database credentials:

python
Copy
Edit
import mysql.connector
conn = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="attendance_system"
)
You may need to manually create tables or import SQL if not automated.

5. Run the Flask App
bash
Copy
Edit
python app.py
6. Open in Browser
Visit: http://localhost:5000

🧠 Facial Recognition Details
Uses InsightFace for accurate and efficient facial embeddings.

Students upload face images for training.

System generates 512-dimension face embeddings stored locally.

Attendance marking compares real-time embedding with trained data using cosine similarity.

📦 Dependencies
Below is a list of packages used in the project:

txt
Copy
Edit
Flask==2.3.3
Werkzeug==2.3.7
insightface==0.7.3
onnxruntime==1.17.3
numpy==1.26.4
pandas==2.2.2
opencv-python==4.9.0.80
mysql-connector-python==8.4.0
Note: If using GPU for faster inference, replace onnxruntime with onnxruntime-gpu.

Install dependencies:

pip install -r requirements.txt
📤 Export Functionality
Admin can export all attendance data to Excel (.xlsx) format using Pandas.

Files are saved to the attendance_records/ folder with timestamps.

📌 Future Enhancements
Role-based access control (multi-admin system)

Email notifications on attendance submission

Real-time dashboard stats and charts

Docker containerization for deployment

REST API support

🙌 Acknowledgements
InsightFace

Flask

OpenCV

MySQL

Pandas
