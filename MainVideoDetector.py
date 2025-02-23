from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
app = Flask(__name__)

# Load the YOLO model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # For YOLOv5


# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Define class names
class_labels = ['Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
                'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage',
                'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
                'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent', 'Number-Plate']


@app.route('/damage/detect', methods=['POST'])
def predict():
    # # Get image from POST request
    # if 'file' not in request.files:
    #     return jsonify({"error": "No file provided"}), 400
    #
    # file = request.files['file']
    #
    # # Convert image to numpy array
    # img_bytes = np.frombuffer(file.read(), np.uint8)
    # image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    concat_number = ''
    number_conf = 0.0

    video_path = "Media/test/vecteezy_broken-cars-after-a-traffic-accident-in-the-parking-lot-of-a_11123202.mov"
    video_capture = cv2.VideoCapture(video_path)
    frame_detections = []
    # Initialize dictionary to store total confidence and count for each label
    confidence_dict = {label: {'total_confidence': 0, 'count': 0} for label in class_labels}

    #frame_skip = 3  # Skip every 3rd frame
    #frame_count = 0
    resize_factor = 0.5

    while True:
        ret, frame = video_capture.read()
        if not ret or len(frame_detections) >= 100:
            break  # Exit the loop if there are no more frames

        # Skip frames
        # if frame_count % frame_skip != 0:
        #     frame_count += 1
        #     continue  # Skip processing this frame

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Perform inference on the current frame
        results = yolo_model.predict(frame)
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
