import cv2
import socket
import csv
import time
import threading
import datetime
import os
import numpy as np
import pyrealsense2 as rs  
from deepface import DeepFace
import argparse
import signal # <-- NEW
import sys    # <-- NEW

# --- DYNAMIC CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="Vision Node Logger")
parser.add_argument("--id", type=str, help="Participant ID")
parser.add_argument("--exp", type=str, help="Experiment Folder Name")
parser.add_argument("--profile", type=str, help="Profile/File Suffix")
args = parser.parse_args()

# --- NEW: CATCH BACKGROUND TERMINATION SIGNALS ---
def handle_shutdown(signum, frame):
    print(f"\n[SYSTEM] Received background shutdown signal. Saving video and exiting...")
    raise KeyboardInterrupt # This forces the script down into your 'finally' cleanup block!

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown) # Catches the kill -15 from Bash

# --- FALLBACK INTERACTIVE INPUTS ---
if args.id is None:
    args.id = input("Enter Participant ID (e.g., P01): ")
if args.exp is None:
    args.exp = input("Enter Experiment Folder Name (e.g., Image Experiment): ")
if args.profile is None:
    args.profile = input("Enter Profile Suffix (e.g., IMAGE_BASELINE): ")

print(f"\n[CONFIG] Loaded -> ID: {args.id} | Exp: {args.exp} | Profile: {args.profile}")


# --- 1. SETUP NETWORK & VARIABLES ---
# Port is now locked to 5005 for all experiments
UDP_IP_LISTEN = "127.0.0.1"
UDP_PORT_LISTEN = 5005

latest_trigger = 0 
trigger_lock = threading.Lock()

# --- 2. UDP LISTENER THREAD ---
def udp_listener():
    global latest_trigger
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP_LISTEN, UDP_PORT_LISTEN))
    print(f"Listening for triggers on {UDP_IP_LISTEN}:{UDP_PORT_LISTEN}...")
    
    while True:
        data, addr = sock.recvfrom(1024)
        if data:
            with trigger_lock:
                latest_trigger = int(data.decode('utf-8'))
                human_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"Trigger {latest_trigger} received at {human_time}!")

listener_thread = threading.Thread(target=udp_listener, daemon=True)
listener_thread.start()

# --- 3. FACE DETECTION AI THREAD ---
frame_for_detection = None
detection_lock = threading.Lock()

detected_emotion = "no face"
detected_box = None
box_lock = threading.Lock()

def face_detector_thread():
    global frame_for_detection, detected_emotion, detected_box
    print("AI Detection Thread Started...")
    while True:
        with detection_lock:
            current_frame = frame_for_detection.copy() if frame_for_detection is not None else None
        
        if current_frame is not None:
            try:
                result = DeepFace.analyze(
                    current_frame, 
                    actions=['emotion'], 
                    enforce_detection=True,
                    # detector_backend='retinaface' 
                    detector_backend='opencv' 
                )
                
                face_data = result[0]
                region = face_data['region']
                w, h = region['w'], region['h']

                if w >= 80 and h >= 80:
                    with box_lock:
                        detected_box = (region['x'], region['y'], w, h)
                        detected_emotion = face_data['dominant_emotion'] 
                else:
                    raise ValueError("Face too small")

            except ValueError:
                with box_lock:
                    detected_box = None
                    detected_emotion = "no face" 
            
            time.sleep(0.01)
        else:
            time.sleep(0.1)

ai_thread = threading.Thread(target=face_detector_thread, daemon=True)
ai_thread.start()

# --- 4. REALSENSE CAMERA SETUP ---
print("Waking up Intel RealSense camera (Single RGB Stream)...")
pipeline = rs.pipeline()
config = rs.config()

frame_width = 640
frame_height = 480
fps = 30

config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)

try:
    pipeline.start(config)
    print("SUCCESS: Connected to RealSense RGB stream!")
except Exception as e:
    print(f"FATAL: Could not start RealSense camera. Error: {e}")
    exit()

# --- 5. SETUP DIRECTORIES, VIDEO WRITER & CSV ---
now = datetime.datetime.now()
date_folder = now.strftime("%Y-%m-%d")  
time_stamp = now.strftime("%H-%M-%S")   

# AUTOMATIC TARGET ROUTING 
base_dir = "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data"
specific_dir = f"{args.id}_{date_folder}/{args.exp}"
save_dir = os.path.join(base_dir, specific_dir)

os.makedirs(save_dir, exist_ok=True)

video_filename = os.path.join(save_dir, f"vision_video_{time_stamp}_{args.profile}.avi")
csv_filename = os.path.join(save_dir, f"vision_log_{time_stamp}_{args.profile}.csv")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
# ADDED 'Experiment_Time' TO CSV HEADER
csv_writer.writerow(['Timestamp', 'Experiment_Time', 'Frame_Count', 'Trigger', 'Emotion'])

print(f"\n✅ Saving vision data to: {save_dir}")
print("Vision Node Started. Recording Video and Log...")

frame_count = 0

# Start the exact elapsed timer!
experiment_start_time = time.time()

try:
    # --- 6. MAIN LOOP ---
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame: 
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        
        with detection_lock:
            frame_for_detection = color_image
        
        frame_count += 1
        
        # Calculate Experiment Time (seconds, rounded to 3 decimal places)
        current_experiment_time = round(time.time() - experiment_start_time, 3)
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        out_video.write(color_image)

        with trigger_lock:
            trigger_to_log = latest_trigger
            latest_trigger = 0 

        with box_lock:
            current_box = detected_box
            current_emotion = detected_emotion

        if current_box is not None:
            x, y, w, h = current_box
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(color_image, f"EMOTION: {current_emotion.upper()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(color_image, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if trigger_to_log != 0:
            cv2.putText(color_image, f"TRIGGER: {trigger_to_log}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Write ALL data to CSV
        csv_writer.writerow([current_time_str, current_experiment_time, frame_count, trigger_to_log, current_emotion])
        
        cv2.imshow('Sisyphus Nervous System: Eyes', color_image)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break

except KeyboardInterrupt:
    print("\nCtrl+C detected in terminal! Forcing a safe shutdown...")

finally:
    # --- 7. CLEAN UP ---
    print("Cleaning up camera resources...")
    pipeline.stop()  
    out_video.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Data saved successfully to {save_dir}.")