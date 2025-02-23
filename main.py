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
                'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent']


@app.route('/predict', methods=['POST'])
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

    image_path = "Media/test/istockphoto-1385753965-640x640.jpg"
    image = cv2.imread(image_path)

    # Perform inference
    results = yolo_model.predict(image)

    print(results)

    # Check if results contain predictions
    if not results:
        return jsonify({"error": "No predictions found"}), 404

    # Convert results to JSON format
    detections = results[0]

    # Ensure detections are available
    if detections is None or len(detections.boxes) == 0:
        return jsonify({"error": "No objects detected"}), 404

    # Parse boxes, classes, and scores
    boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    scores = detections.boxes.conf.cpu().numpy()  # Confidence scores
    classes = detections.boxes.cls.cpu().numpy()  # Class IDs

    # Create a structured response
    predictions = []
    for box, score, cls in zip(boxes, scores, classes):
        predictions.append({
            'class_name': class_labels[int(cls)],
            'confidence': float(score),  # Convert to float for JSON serialization
            'bbox': {
                'xmin': int(box[0]),
                'ymin': int(box[1]),
                'xmax': int(box[2]),
                'ymax': int(box[3])
            }
        })

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)