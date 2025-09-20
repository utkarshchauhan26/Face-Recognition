# 👤 Real-Time Face Recognition with PyTorch & OpenCV

A real-time face detection and recognition application that:
- Captures live webcam video
- Detects faces using **MTCNN**
- Generates embeddings with **InceptionResnetV1**
- Matches detected faces against a known database of images

---

## 🚀 Features
- **Real-Time Detection**: Continuously detects faces from your webcam feed.
- **Face Encoding**: Generates 512-dimensional embeddings for accurate recognition.
- **Customizable**: Easily add more known faces by adding images to the `Images/` folder and updating the dictionary.

---

## 🗂 Project Structure
.
├── Images/
│ ├── utkarsh.jpg
│ └── shah.jpg
├── requirements.txt
└── face_recognition_app.py

yaml
Copy code

---

## 🛠 Tech Stack
- **Python 3.x**
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- NumPy

---

## ⚡️ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
Create a requirements.txt with:

nginx
Copy code
torch
facenet-pytorch
opencv-python
numpy
3️⃣ Add Known Faces
Place images of known people inside the Images/ folder.

Update the known_faces dictionary in face_recognition_app.py:

python
Copy code
known_faces = {
    "Name1": "Images/name1.jpg",
    "Name2": "Images/name2.jpg"
}
4️⃣ Run the App
bash
Copy code
python face_recognition_app.py
Press q to exit the webcam window.

⚙️ Configuration
Threshold: Adjust the similarity threshold in recognize_faces (default: 0.6) for stricter or looser matching.

Camera Source: Change cv2.VideoCapture(0) to another index or video path if needed.

📸 Demo
When a known face is recognized, a green rectangle and the person’s name appear in real time on the webcam feed.

🛡 License
MIT License – feel free to use and modify.

💡 Future Improvements
Store embeddings in a database for faster startup.

Add GUI controls to register new users on the fly.

Implement multi-camera support.

