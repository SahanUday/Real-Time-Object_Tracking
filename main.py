from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolo11n.pt')

current_camera_index = 0  # Track the active webcam index

# Open the webcam (0 is the default camera index, adjust if necessary)
cap = cv2.VideoCapture(current_camera_index)

# Create a resizable window
cv2.namedWindow('Real-Time Person Detection', cv2.WINDOW_NORMAL)
is_fullscreen = False  # To toggle fullscreen mode

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Extract detections for persons only
    person_boxes = []
    for result in results:
        for box in result.boxes:
            # Check if the class corresponds to 'person' (class ID 0 for COCO)
            if box.cls == 0:  # cls represents the class index
                person_boxes.append(box.xyxy)  # Append bounding box coordinates

    # Draw bounding boxes for detected persons
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[0])  # Extract and convert coordinates to integers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Person Detection', frame)

    # Handle keypress events (used to handle keyboard events in OpenCV applications)
    key = cv2.waitKey(1) & 0xFF

    # Quit on 'q'
    if key == ord('q'):
        break

    # Switch cameras on 's'
    if key == ord('s'):
        print("Switching cameras...")
        cap.release()  # Release the current camera
        current_camera_index = (current_camera_index + 1) % 2  # Toggle between 0 and 1
        cap = cv2.VideoCapture(current_camera_index)

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
