from flask import Flask, render_template_string, Response
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Initialize Flask application
app = Flask(__name__)

# Initialize the video capture (0 is typically the default webcam)
# NOTE: This requires a local webcam to be connected and available.
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    cap = None

# Initialize the YOLOv8 model and annotation tools
# NOTE: This requires the 'yolov8n.pt' model file to be present in the same directory or accessible.
try:
    model = YOLO('yolov8n.pt')
    box_annotator = sv.BoxAnnotator(thickness=2)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None
    box_annotator = None


def detect_and_avoid(frame):
    """
    Performs object detection and applies the Tron bot avoidance logic on a single video frame.
    
    Args:
        frame (np.array): The input video frame from the webcam.

    Returns:
        np.array: The processed frame with annotations and bot logic visualization.
    """
    # Check if the model is loaded before attempting detection
    if model is None:
        # If model is not available, just return the original frame
        return frame

    # Run YOLOv8 detection on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.tolist()
    
    labels = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        # Check if the class name exists in the model's names dictionary
        if cls in model.model.names:
            label = model.model.names[cls]
            labels.append(f"{label} {conf:.2f}")

            # Draw simple avoidance logic line on the frame
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            # OpenCV uses BGR color order
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, label, (cx, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Simulated "Tron" bot logic
    height, width = frame.shape[:2]
    bot_x, bot_y = width//2, height-30
    # Check if any "person" label is present in the detected labels
    safe = not any("person" in l.lower() for l in labels)

    status_color = (0, 255, 0) if safe else (0, 0, 255) # Green for safe, Red for obstacle
    status = "Path Clear - Moving Forward" if safe else "Obstacle Ahead - Turning"

    # Draw the bot's movement arrow
    if safe:
        cv2.arrowedLine(frame, (bot_x, bot_y), (bot_x, bot_y - 50), status_color, 5)
    else:
        cv2.arrowedLine(frame, (bot_x, bot_y), (bot_x + 50, bot_y), status_color, 5)

    # Put the status text on the frame
    cv2.putText(frame, status, (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def generate_frames():
    """
    Generator function to create a motion JPEG stream from the webcam feed.
    """
    # Check if the webcam is open
    if cap is None or not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nWebcam not available\r\n'
        return

    while True:
        success, frame = cap.read()
        if not success:
            # If frame capture fails, break the loop
            break
        
        # Process the frame with the detection and avoidance logic
        processed_frame = detect_and_avoid(frame)
        
        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the frame as part of the multipart stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """
    The main route that renders the HTML page with the video stream.
    """
    # HTML template to display the video feed
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLOv8 Tron Bot Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', sans-serif;
            }
        </style>
    </head>
    <body class="bg-gray-900 text-white flex flex-col items-center justify-center min-h-screen p-4">

        <div class="bg-gray-800 p-8 rounded-2xl shadow-xl max-w-4xl w-full">
            <h1 class="text-4xl font-bold text-center mb-2 text-blue-400">YOLOv8 Tron Bot Live Feed</h1>
            <p class="text-center text-gray-400 mb-6">
                Live stream from your webcam with object detection and avoidance logic.
            </p>

            <!-- Image tag pointing to the video feed route -->
            <div class="relative w-full aspect-video rounded-xl overflow-hidden shadow-2xl border-4 border-gray-700">
                <img src="{{ url_for('video_feed') }}" class="w-full h-full" alt="YOLOv8 Live Feed">
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """
    This route streams the video frames from the webcam.
    It returns a Response object with a multipart content type.
    """
    # Return the response with the generator function and the correct mimetype
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the Flask app on the local network interface
    app.run(host='0.0.0.0', port=5000, debug=True)
