from flask import Flask, render_template, Response
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import os
from flask import Flask, redirect, render_template, request, Response
from werkzeug.utils import secure_filename
from flask import Flask, redirect, render_template, request, Response
app = Flask(__name__)
video_path = None
 # For Video
# Define the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model = YOLO("../Yolo-Weights/yolov8l.pt")
class ObjectDetector:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"
                    ]
        self.class_id_to_track = 0

        # Create VideoCapture object
        self.cap = cv2.VideoCapture(video_path)

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter('output_video.mp4', self.fourcc, 30.0, (self.frame_width, self.frame_height))



    def detect_objects(self):
            prev_frame_time = 0
            while True:
                new_frame_time = time.time()
                success, img = self.cap.read()
                if not success:
                    break  # Break the loop when the video ends

                results = model(img, stream=True)
                person_count = 0
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])

                        if cls == self.class_id_to_track:
                            # Bounding Box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1
                            cvzone.cornerRect(img, (x1, y1, w, h))
                            # Confidence
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            # Class Name
                            cls = int(box.cls[0])
                            person_count += 1

                            cvzone.putTextRect(img, f'{self.classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                text = f"Persons: {person_count}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = (img.shape[0] + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.output_video.write(img)  # Write the frame to the output video
                _, frame = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            video_path = video_path
            print(video_path)
            # return Response(ObjectDetector( video_path ,"../Yolo-Weights/yolov8l.pt").detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')
            response = Response(ObjectDetector(video_path ,"../Yolo-Weights/yolov8l.pt").detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')
            response.headers['Refresh'] = '10;url=/another_page'  # Redirect to /another_page after 5 seconds
            return response
            # session['redirect_url'] = '/another_page'
            # return Response(ObjectDetector(video_path, "../Yolo-Weights/yolov8l.pt").detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

        

         

    return render_template('upload.html')

@app.route('/another_page')
def another_page():
    return render_template('another_page.html')
if __name__ == '__main__':
    app.run(debug=True, port=5002)
