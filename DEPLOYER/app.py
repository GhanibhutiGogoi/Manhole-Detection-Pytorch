import base64
from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import io
import supervision as sv
from threading import Thread
torch.cuda.is_available()
HTML_TEMPLATE = r'''
 <!DOCTYPE html>
    <html>
    <head>
        <title>Live Image Classification</title>
    </head>
    <body>
        <video id="video" width="480" height="640" autoplay></video>
        <img id="result" src="" alt="Processed Image" />
        <script>
            async function setupCamera() {
                const video = document.getElementById('video');
                const stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {exact: 'environment'}}});
                video.srcObject = stream;
            }

            function captureAndSendFrame() {
                const video = document.getElementById('video');
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    const formData = new FormData();
                    formData.append('image', blob);

                    fetch('/classify', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.blob())
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        document.getElementById('result').src = url;
                    })
                    .catch(error => {
                        console.error('Fetch error:', error);
                    });
                }, 'image/jpeg');
            }

            setupCamera().then(() => {
                setInterval(captureAndSendFrame, 250);  // Adjust the interval as needed
            });
        </script>
    </body>
    </html>

'''
deploy_Model = YOLO("D:/PYTHON/Manhole Cover Detection/runs/detect/train6/weights/best.pt")
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
box_annot = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
app = Flask(__name__)

def convert_image_to_mat(image_data):
    # Decode image data to numpy array
    np_img = np.frombuffer(image_data, np.uint8)
    # Convert numpy array to OpenCV Mat
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img
# def mat_to_base64_str(mat_img):
#     _, buffer = cv2.imencode(".png", mat_img)
#     base64_str = base64.b64encode(buffer).decode('utf-8')
#     return base64_str
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)
@app.route('/classify', methods=['POST'])
def classify():
    global deploy_Model, box_annot
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    mat_image = convert_image_to_mat(image)
    frame = mat_image
    result = deploy_Model(frame)[0]
    detection = sv.Detections.from_yolov8(result)
    labels = [
        f"{deploy_Model.names[class_id]} {confidence:0.01f}"
        for _, confidence, class_id, _
        in detection
    ]
    frame = box_annot.annotate(scene=frame, detections=detection, labels=labels)
    _, buffer = cv2.imencode('.jpg', frame)
    io_buf = io.BytesIO(buffer)
    last_processed_image = frame
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('D:\PYTHON\Manhole Cover Detection\DEPLOYER\cert.pem', 'D:\PYTHON\Manhole Cover Detection\DEPLOYER\key.pem'))
