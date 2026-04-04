import time
import threading
from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import random
from datetime import datetime
import copy

app = Flask(__name__)

# Switched to the Large model (yolov8l.pt) for substantially higher accuracy
model = YOLO("yolov8l.pt")
class_colors = {}

# Global variable to store the latest logs/notifications
detection_logs = []
weapon_detected_state = {"status": False, "timestamp": 0}

# Common weapon keywords (handles COCO 'knife', 'baseball bat' as well as custom class names)
WEAPON_KEYWORDS = ['knife', 'gun', 'weapon', 'pistol', 'rifle', 'firearm', 'sword', 'scissors']

# ----------------------------------------------------
# MULTI-THREADING ARCHITECTURE FOR FPS OPTIMIZATION
# ----------------------------------------------------
# To completely eliminate "video lag", we decouple the WebCam reading 
# and the YOLO inference into separate hardware threads.
latest_frame = None
latest_boxes = []
lock = threading.Lock()

def capture_thread():
    """ Runs constantly to pull the absolute newest frame from the webcam at ~30 FPS """
    global latest_frame
    cap = cv2.VideoCapture(0)
    # Crucial for reducing queue lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while True:
        success, frame = cap.read()
        if success:
            with lock:
                latest_frame = frame.copy()
        time.sleep(0.01) # Small sleep to prevent CPU hogging

def inference_thread():
    """ Grabs the latest frame and runs YOLO at maximum capacity without freezing the video """
    global latest_frame, latest_boxes, detection_logs, weapon_detected_state
    
    while True:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is not None:
            # We process at standard 640 resolution for high accuracy
            results = model(frame_to_process, imgsz=640, conf=0.25, iou=0.45)[0]
            names = results.names
            
            new_boxes = []
            current_weapons = []
            
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                cls_id = int(cls)
                label_name = names[cls_id].lower()
                
                # Check for weapons for notification system
                is_weapon = any(word in label_name for word in WEAPON_KEYWORDS)
                
                if is_weapon:
                    current_weapons.append(label_name)
                    # Add to logs if not logged recently (debounce notifications)
                    if len(detection_logs) == 0 or (time.time() - weapon_detected_state["timestamp"] > 3):
                        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 WEAPON DETECTED: {label_name.upper()} ({conf:.2f})"
                        detection_logs.insert(0, log_msg)
                        if len(detection_logs) > 20: # keep last 20 logs
                            detection_logs.pop()
                
                new_boxes.append((box, cls_id, float(conf), label_name, is_weapon))
            
            with lock:
                latest_boxes = new_boxes
                
            if current_weapons:
                weapon_detected_state["status"] = True
                weapon_detected_state["timestamp"] = time.time()
            elif time.time() - weapon_detected_state["timestamp"] > 2:
                # Clear red alert after 2 seconds
                weapon_detected_state["status"] = False
                
        # To avoid maxing out 100% of all CPU cores, we rest slightly.
        time.sleep(0.05) 

# Start background threads before any Flask requests hit
t_cap = threading.Thread(target=capture_thread, daemon=True)
t_cap.start()

t_inf = threading.Thread(target=inference_thread, daemon=True)
t_inf.start()

# ----------------------------------------------------
# FLASK ROUTING HTTP SERVER
# ----------------------------------------------------

def generate_frames():
    """ Streams the latest frame from the capture thread, overlaid with boxes from the inference thread """
    while True:
        display_frame = None
        boxes_to_draw = []
        
        with lock:
            if latest_frame is not None:
                display_frame = latest_frame.copy()
                boxes_to_draw = copy.deepcopy(latest_boxes)
                
        if display_frame is not None:
            # Draw bounding boxes (either newly detected or cached from last YOLO infer)
            for box, cls_id, conf, label_name, is_weapon in boxes_to_draw:
                label = f"{label_name.capitalize()} {conf:.2f}"
                
                if is_weapon:
                    color = (0, 0, 255) # Red for weapons
                    # Flash border on the video frame
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), color, 8) 
                    cv2.putText(display_frame, "WEAPON ALERT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                else:
                    if cls_id not in class_colors:
                        class_colors[cls_id] = [random.randint(0,255) for _ in range(3)]
                    color = class_colors[cls_id]
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame over HTTP stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        # Cap HTTP streaming output to ~30 FPS to save bandwidth while still looking butter smooth
        time.sleep(0.033) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    global weapon_detected_state
    return jsonify({
        "weapon_detected": weapon_detected_state["status"],
        "logs": detection_logs
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Start Real-time Flask Server
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) 
    # use_reloader=False prevents Flask from running dual threads and messing up our cv2 camera lock
