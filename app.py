from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        result_image = detect_faces(image)
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image = base64.b64encode(buffer).decode('utf-8')
        return render_template('index.html', result_image=result_image)
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_from_directory(os.path.join(app.root_path, '..', 'Assets'), filename)

if __name__ == '__main__':
    app.run(debug=True)