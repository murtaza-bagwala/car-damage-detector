import base64
import tempfile

import cv2
import requests
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load your YOLO model (replace with your model's path)
model = YOLO("Weights/best.pt")

class_labels = ['Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
                'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage',
                'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
                'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent', 'Number-Plate']

local_video_file = 'downloaded_video.mp4'

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    video_url = data.get('video_url')
    print("Downloading video...")
    download_video(video_url, local_video_file)
    print(f"Video downloaded and saved to {local_video_file}")

    # if not base64_video:
    #     return jsonify({"error": "No video provided"}), 400

    # # Decode the Base64 string and write it to a temporary video file
    # video_data = base64.b64decode(base64_video)
    #
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
    #     temp_video_file.write(video_data)
    #     temp_video_path = temp_video_file.name
    # Open the video file (replace with your video file path)
    #video_path = 'Media/test/recording.mov'
    cap = cv2.VideoCapture(local_video_file)

    # Frame skipping factor (adjust as needed for performance)
    frame_skip = 3  # Skip every 3rd frame
    frame_count = 0
    concat_number = ''
    number_conf = 0.0

    while cap.isOpened():
        if number_conf > 0.60:
            break
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Exit loop if there are no frames left

        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue  # Skip processing this frame

        # Resize the frame (optional, adjust size as needed)
        frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

        # Make predictions on the current frame
        results = model.predict(source=frame)

        # Iterate over results and draw predictions
        for result in results:
            boxes = result.boxes  # Get the boxes predicted by the model
            for box in boxes:
                class_id = int(box.cls)  # Get the class ID
                confidence = box.conf.item()  # Get confidence score
                coordinates = box.xyxy[0]  # Get box coordinates as a tensor

                # Extract and convert box coordinates to integers
                x1, y1, x2, y2 = map(int, coordinates.tolist())  # Convert tensor to list and then to int

                # Draw the box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle

                # Try to apply OCR on detected region
                try:
                    # Ensure coordinates are within frame bounds
                    r0 = max(0, x1)
                    r1 = max(0, y1)
                    r2 = min(frame.shape[1], x2)
                    r3 = min(frame.shape[0], y2)

                    # Crop license plate region
                    plate_region = frame[r1:r3, r0:r2]

                    # Convert to format compatible with EasyOCR
                    plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                    plate_array = np.array(plate_image)

                    # Use EasyOCR to read text from plate
                    plate_number = reader.readtext(plate_array)
                    concat_number = ' '.join([number[1] for number in plate_number])
                    number_conf = np.mean([number[2] for number in plate_number])
                    if number_conf >= 0.59:
                        break

                    # # # Draw the detected text on the frame
                    # cv2.putText(
                    #     img=frame,
                    #     text=f"Plate: {concat_number} ({number_conf:.2f})",
                    #     org=(r0, r1 - 10),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.7,
                    #     color=(0, 0, 255),
                    #     thickness=2
                    # )

                except Exception as e:
                    print(f"OCR Error: {e}")
                    pass





    # Combine all results for the response
    response = {
        'number_plate': concat_number
    }

    return jsonify(response)

@app.route('/damage/detect', methods=['POST'])
def detect_damage():
    # # Get image from POST request
    # if 'file' not in request.files:
    #     return jsonify({"error": "No file provided"}), 400
    #
    # file = request.files['file']
    #
    # # Convert image to numpy array
    # img_bytes = np.frombuffer(file.read(), np.uint8)
    # image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # data = request.json
    # video_url = data.get('video_url')
    # print("Downloading video...")
    # download_video(video_url, local_video_file)
    # print(f"Video downloaded and saved to {local_video_file}")

    concat_number = ''
    number_conf = 0.0

    #video_path = "Media/test/vecteezy_broken-cars-after-a-traffic-accident-in-the-parking-lot-of-a_11123202.mov"
    video_capture = cv2.VideoCapture(local_video_file)
    frame_detections = []
    # Initialize dictionary to store total confidence and count for each label
    confidence_dict = {label: {'total_confidence': 0, 'count': 0} for label in class_labels}

    frame_skip = 3  # Skip every 3rd frame
    frame_count = 0
    resize_factor = 0.5

    while True:
        ret, frame = video_capture.read()
        if not ret or len(frame_detections) >= 100:
            break  # Exit the loop if there are no more frames

        #Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue  # Skip processing this frame

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Perform inference on the current frame
        results = model.predict(frame)
        # Initialize a list to store detections for each frame

        # Extract detections from the results
        if len(results) > 0:
            detections = results[0]

            if detections is not None and len(detections.boxes) > 0:
                boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                scores = detections.boxes.conf.cpu().numpy()  # Confidence scores
                classes = detections.boxes.cls.cpu().numpy()  # Class IDs

                # Store detections for this frame
                frame_detections.append({
                    'frame': video_capture.get(cv2.CAP_PROP_POS_FRAMES),  # Current frame number
                    'detections': []
                })

                for box, score, cls in zip(boxes, scores, classes):
                    class_name = class_labels[int(cls)]
                    frame_detections[-1]['detections'].append({
                        'class_name': class_labels[int(cls)],
                        'confidence': float(score),  # Convert to float for JSON serialization
                        'bbox': {
                            'xmin': int(box[0]),
                            'ymin': int(box[1]),
                            'xmax': int(box[2]),
                            'ymax': int(box[3])
                        }
                    })

                    # Update total confidence and count for this class
                    confidence_dict[class_name]['total_confidence'] += float(score)
                    confidence_dict[class_name]['count'] += 1

    # Release video capture
    video_capture.release()

    # Calculate average confidence for each class
    average_confidence = {label: 0 for label in class_labels}
    for label in class_labels:
        if confidence_dict[label]['count'] > 0:
            average_confidence[label] = (
                    confidence_dict[label]['total_confidence'] /
                    confidence_dict[label]['count']
            )

    # Combine all results for the response
    response = {
        'frame_detections': frame_detections,
        'average_confidence_per_label': average_confidence,
        'nuber_plate': f"Plate: {concat_number} ({number_conf:.2f})"
    }


    return jsonify(response)

def download_video(url, local_filename):
    """Download video from a URL and save it locally."""
    # Stream the download to avoid loading whole file into memory
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for any error responses
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # Write in chunks
                f.write(chunk)
    return local_filename


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

