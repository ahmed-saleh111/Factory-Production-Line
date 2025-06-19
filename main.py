import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
# Load the YOLO model
yolo_model = YOLO("best.pt")
names = yolo_model.names

# Open the video file
cap = cv2.VideoCapture('vid4.mp4')
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Constants for ROI detection and tracking
cx1 = 491
offset = 8

# Get current date for folder and file naming
current_date = time.strftime("%Y-%m-%d")

# Create a folder for cropped images
crop_folder = f"crop_{current_date}"
os.makedirs(crop_folder, exist_ok=True)

# Set to track processed track_ids
processed_track_ids = set()

def encode_image_to_base64(image):
    """Convert an image to a base64 string."""
    _, img_buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(img_buffer).decode('utf-8')

def analyze_image_with_openai(current_image):
    """Analyze a single image using OpenRouter's OpenAI."""
    if current_image is None:
        return "No image available for analysis."

    # Convert image to base64
    current_image_data = encode_image_to_base64(current_image)

    message = HumanMessage(
        content = [
            {'type': 'text', 
             'text': "Analyze this image and determine if the label is present on the bottle. Check the following:\n\n"
                                  "1. **Is the label present?** (Yes/No)\n"
                                  "2. **Is there any damage?** (Yes/No)\n\n"
                                  "Return the result strictly in a structured table format like below:\n\n"
                                  "| Label Present | Damage |\n"
                                  "|--------------|--------|\n"
                                  "| Yes/No       | Yes/No |"

            },
            {'type': 'image_url', 
             'image_url': {'url': f"data:image/jpeg;base64,{current_image_data}"}}

        ]
    )
    
    try:
        response = model.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error invoking model: {e}")
        return "Error processing image."

def save_response_to_file(track_id, response, crop_filename):
    """Save the analysis response to a separate .txt file for each bottle."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract the base filename from the cropped image
    txt_filename = crop_filename.replace(".jpg", ".txt")
    
    # Create the file and save the structured response
    with open(txt_filename, "w", encoding="utf-8") as file:
        file.write(f"Track ID: {track_id}\n")
        file.write(f"Date: {timestamp}\n")
        file.write(f"{response}\n")

def save_crop_image(crop, track_id):
    """Save cropped image with track ID."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{crop_folder}/{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, crop)
    return filename

def crop_and_process(frame, box, track_id):
    """Crop detected objects and send for analysis."""
    if track_id in processed_track_ids:
        return  # Skip if already processed

    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    # Save the cropped image
    crop_filename = save_crop_image(crop, track_id)

    # Mark as processed
    processed_track_ids.add(track_id)

    # Start analysis in a separate thread
    threading.Thread(target=process_crop_image, args=(crop, track_id, crop_filename)).start()

def process_crop_image(current_image, track_id, crop_filename):
    """Process the cropped image and analyze it using OpenAI."""
    response_content = analyze_image_with_openai(current_image)
    print(f"OpenAI Response for Track ID {track_id}:", response_content)
    
    # Save response in a separate .txt file for each bottle
    save_response_to_file(track_id, response_content, crop_filename)

def process_video_frame(frame):
    """Process video frame for object detection and analysis."""
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking
    results = yolo_model.track(frame, persist=True)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx = int(x1 + x2) // 2

            if cx1 < (cx + offset) and cx1 > (cx - offset):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                crop_and_process(frame, box, track_id)

    return frame

def main():
    """Main function to run video processing."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_video_frame(frame)

        cv2.line(frame, (491, 1), (491, 499), (0, 0, 255), 2)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
