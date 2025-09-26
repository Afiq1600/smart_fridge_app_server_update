from fastapi import FastAPI, WebSocket
import uvicorn
import cv2
import argparse
import os
import json
import random
import numpy as np
from collections import deque, defaultdict
import time
from threading import Thread
import threading
import RPi.GPIO as GPIO
import atexit
import requests
from requests.auth import HTTPBasicAuth
import asyncio
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib,GObject
import hailo
import multiprocessing
import setproctitle
import io
from hailo_pipeline import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    get_default_parser,
    detect_hailo_arch,
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE
)
from typing import List, Dict, Tuple
import signal
import threading
from multiprocessing import Event
from datetime import datetime
# Initialize FastAPI app
app = FastAPI()
data_deque: Dict[int, deque] = {}


# GPIO Pin definitions (kept from original)
DOOR_LOCK_PIN = 25
DOOR_SWITCH_PIN = 26
LED_GREEN = 23
LED_RED = 18
BUZZER_PIN = 20

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH) # Initial Buzzer Tak Bunyi
GPIO.setup(LED_GREEN, GPIO.OUT, initial=GPIO.HIGH) # Initial Green Mati
GPIO.setup(LED_RED, GPIO.OUT, initial=GPIO.HIGH) #Initial Red Hidup
GPIO.setup(DOOR_LOCK_PIN, GPIO.OUT, initial=GPIO.HIGH) # Initial Lock
GPIO.setup(DOOR_SWITCH_PIN, GPIO.IN)

# Initialize tracker globally
readyToProcess = False
blink = False
alert_thread = None


camera_covered = False
cover_alert_thread = None
# Store movement history for each tracked object
movement_history = defaultdict(lambda: deque(maxlen=5))  # Store last 5 positions for each track_id
movement_direction = {}  # Store calculated direction for each track_id
last_counted_direction = {}  # Store the last counted direction for each track_id
def trigger_buzzer(duration=0.5):
    """
    Trigger the buzzer for a specified duration
    Args:
        duration (float): Duration in seconds to keep the buzzer on
    """
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)


def blink_led(pin, times, delay):
    """
    Blink an LED a specified number of times with a delay.
    
    Args:
        pin: GPIO pin connected to the LED
        times: Number of times the LED should blink
        delay: Delay in seconds between ON and OFF states
    """
    for _ in range(times):
        GPIO.output(pin, GPIO.HIGH)  # Turn LED on
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)  # Turn LED off
        time.sleep(delay)
          

def control_door(pin, action, duration=0.5):
    """
    Control the door lock mechanism.
    
    Args:
        pin: GPIO pin connected to the door lock mechanism
        action: String indicating 'lock' or 'unlock'
        duration: Time in seconds to keep the door unlocked (default: 3 seconds)
    """
    if action.lower() == 'unlock':
        print("Unlocking door...")
        GPIO.output(pin, GPIO.LOW)    # Activate the lock mechanism (unlock)
        time.sleep(duration)          # Keep unlocked for specified duration
        GPIO.output(pin, GPIO.HIGH)   # Deactivate (return to locked state)
        print("Door locked again")
    elif action.lower() == 'lock':
        print("Locking door...")
        GPIO.output(pin, GPIO.HIGH)   # Ensure the door is locked
        print("Door locked")
    else:
        print("Invalid action. Use 'lock' or 'unlock'")
        
        
def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)





# Define global trail storage at the beginning of your file
object_trails = defaultdict(lambda: deque(maxlen=30))
global_trails = defaultdict(lambda: deque(maxlen=30))

def draw_trail(frame, track_id, center, color, global_id=None):
    """
    Draw movement trail for a tracked object
    Args:
        frame: Frame to draw on
        track_id: Local track ID or global ID
        center: Current center point (x, y)
        color: Color for drawing
        global_id: Global ID for this track (if available)
    """
    # If global_id is provided, use global trails
    if global_id is not None:
        global_trails[global_id].appendleft(center)
        points = list(global_trails[global_id])
    else:
        # Use track_id as the key (could be local or global ID)
        object_trails[track_id].appendleft(center)
        points = list(object_trails[track_id])
    
    # Draw the trail
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2)
        cv2.line(frame, points[i - 1], points[i], color, thickness)

def draw_counts(frame, class_counters, label):
    
    class_names = {
    0: "",
    1: "bottle",
 
}
    
    """Draw both entry and exit counts on frame"""
    # Calculate totals
    total_entry = sum(class_counters["entry"].values())
    total_exit = sum(class_counters["exit"].values())
    
    # Draw total counts
    cv2.putText(frame, f'Total Entry: {total_entry}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Exit: {total_exit}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw class-specific counts (combined entry and exit on same line)
    y_offset = 110  # Starting y position for class counts
    
    # Get all unique labels from both entry and exit counters
    all_labels = set(class_counters["entry"].keys()) | set(class_counters["exit"].keys())
    
    for label in all_labels:
        entry_count = class_counters["entry"].get(label, 0)
        exit_count = class_counters["exit"].get(label, 0)
        
        class_id = next(k for k, v in class_names.items() if v == label)
        color = compute_color_for_labels(class_id)
        
        text = f'{label} Entry: {entry_count}, Exit: {exit_count}'
        cv2.putText(frame, text, (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

    
   
        


def draw_zone(frame):
    """Draw a single zone covering the entire frame"""
    height, width = frame.shape[:2]
    # Draw rectangle around the entire frame
    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)





def handle_alert_state():
    global blink
    while blink:
        if GPIO.input(DOOR_SWITCH_PIN) == 0:  # Check if door is closed
            GPIO.output(LED_RED, GPIO.LOW)  # Turn LED off
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer off
            break
        GPIO.output(LED_RED, GPIO.HIGH)  # Turn LED on
        time.sleep(0.5)
        GPIO.output(LED_RED, GPIO.LOW)   # Turn LED off
        time.sleep(0.5)

def calculate_total_price_and_control_buzzer(current_data, deposit):
    """
    Calculate total price for validated items and control buzzer based on deposit comparison
    """
    global blink, alert_thread, price_alert_sound_playing
    total_product_price = 0
    
    # Process validated products
    validated_products = current_data.get("validated_products", {})
    all_products = set(validated_products.get("entry", {}).keys()) | set(validated_products.get("exit", {}).keys())
    
    for product_name in all_products:
        entry_data = validated_products.get("entry", {}).get(product_name, {"count": 0})
        exit_data = validated_products.get("exit", {}).get(product_name, {"count": 0})
        
        entry_count = entry_data.get("count", 0)
        exit_count = exit_data.get("count", 0)
        
        product_details = exit_data.get("product_details") or entry_data.get("product_details")
        if product_details and "product_price" in product_details:
            price_per_unit = float(product_details["product_price"])
            true_count = max(0, exit_count - entry_count)
            total_product_price += true_count * price_per_unit
    
    # Control buzzer, LED, and sound based on price comparison
    if total_product_price > deposit:
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer on
        blink = True
        
        # Start price alert sound if not already playing
        if not price_alert_sound_playing:
            price_alert_sound_playing = True
            tts_manager.play_mp3_async("sounds/siren1.mp3", volume=1.0)
            print(f"Price alert: ${total_product_price:.2f} > ${deposit:.2f} - playing alert sound")
        
        # Start LED blinking in a new thread if not already running
        if alert_thread is None or not alert_thread.is_alive():
            alert_thread = threading.Thread(target=handle_alert_state, daemon=True)
            alert_thread.start()
    else:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer off
        blink = False
        GPIO.output(LED_RED, GPIO.LOW)  # Ensure LED is off
        
        # Stop price alert sound if it was playing
        if price_alert_sound_playing:
            price_alert_sound_playing = False
            # Only stop audio if camera is not covered (to avoid stopping camera alert)
            if not camera_covered_sound_playing:
                tts_manager.stop_all_audio()
            print("Price within deposit limit - stopping price alert sound")
    
    return total_product_price

print_lock = threading.Lock()
def check_door_status():
    """Continuously monitor door switch status"""
    while True:
        door_sw = 1
        with print_lock:
         if door_sw == 0:  # Door is closed
            print("Door closed - Shutting down preview frames")
            return True
         time.sleep(0.1)  # Small delay to prevent CPU overuse

import subprocess




# Add these global variables at the top of your file
camera_covered_sound_playing = False
price_alert_sound_playing = False

def is_frame_dark(frame, threshold=40):
    """
    Check if the frame is mostly dark (covered)
    Args:
        frame: The input frame
        threshold: Brightness threshold (0-255)
    Returns:
        bool: True if frame is dark, False otherwise
    """
    global camera_covered_sound_playing
    
    # Convert frame to grayscale if it's not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    is_dark = avg_brightness < threshold
    
    # Handle sound playback for camera coverage
    if is_dark and not camera_covered_sound_playing:
        # Start playing camera covered alert sound
        camera_covered_sound_playing = True
        tts_manager.play_mp3_async("sounds/siren1.mp3", volume=1.0)
        print("Camera covered - playing alert sound")
    elif not is_dark and camera_covered_sound_playing:
        # Stop camera covered alert sound
        camera_covered_sound_playing = False
        tts_manager.stop_all_audio()
        print("Camera uncovered - stopping alert sound")
    
    return is_dark


def handle_cover_alert():
    """Handle buzzer alert when camera is covered"""
    global camera_covered
    while camera_covered:
        if GPIO.input(DOOR_SWITCH_PIN) == 0:  # Check if door is closed
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer off
            break
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer on
        #time.sleep(0.1)  # Short sleep to allow for more responsive checking
    GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Ensure buzzer is off when exiting


def display_user_data_frame(user_data):
    """Display frames from user data, monitor door status, and save video to OS"""
    # Start door monitoring in a separate thread
    door_monitor_thread = threading.Thread(target=check_door_status)
    door_monitor_thread.daemon = True  # Thread will exit when main program exits
    door_monitor_thread.start()
    
    # Get transaction details from user_data
    transaction_id = getattr(user_data, 'transaction_id', None) 
    machine_id = getattr(user_data, 'machine_id', None) 
    user_id = getattr(user_data, 'user_id', None) 
    machine_identifier = getattr(user_data, 'machine_identifier', None)
    
    # Create video directory if it doesn't exist
    video_dir = os.path.join(os.getcwd(), "saved_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Set up video writer to save to file system
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = None
    
    # Generate filename for saving to OS
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    dataset_name = f"hailo_detection_{timestamp}_{transaction_id}"
    filename = os.path.join(video_dir, f"{dataset_name}.avi")
    
    try:
        while not user_data.shutdown_event.is_set():
            # Check if door is closed
            door_sw = 1
            
            frame = user_data.get_frame()
            if frame is not None:
                # Create video writer on first frame to get dimensions
                if output_video is None:
                    height, width = frame.shape[:2]
                    output_video = cv2.VideoWriter(filename, fourcc, 25.0, (width, height), isColor=True)
                    print(f"Started recording to: {filename}")
                
                # Write frame to video
                output_video.write(frame.copy())
                
                cv2.imshow("Hailo Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Error in display loop: {e}")
    finally:
        print("Cleaning up display resources...")
        
        # Release video writer
        if output_video is not None:
            output_video.release()
            print(f"Video saved successfully to: {filename}")
        
        try:
            # Clean GPIO
            GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
            GPIO.output(LED_GREEN, GPIO.HIGH)
            GPIO.output(LED_RED, GPIO.HIGH)
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        for i in range(5):  # Force windows to close
            cv2.waitKey(1)
            
        # Set shutdown event last
        user_data.shutdown_event.set()
        print("Display cleanup complete")


def stream_video_to_api(video_path, dataset_name, transaction_id, machine_id, user_id, machine_identifier):
    """Stream the video directly to the API endpoint"""
    # API endpoint
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/shopping_app/machine/TransactionDataset/insert_transactionDataset"
    
    # Authentication
    username = 'admin'
    password = '1234'
    api_key = '123456'
    
    # Get current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract filename from path
    filename = os.path.basename(video_path)
    
    # Prepare payload
    payload = {
        'machine_id': machine_id,
        'created_by': user_id,
        'dataset_url': f"assets/video/machine_transaction_dataset/{machine_identifier}/{dataset_name}.avi",
        'dataset_name': dataset_name,
        'transaction_id': transaction_id,
        'created_datetime': current_time
    }
    
    # Prepare headers
    headers = {'x-api-key': api_key}
    
    print(f"Streaming video to API: {video_path}")
    print(f"Payload: {payload}")
    
    try:
        # Open the file in streaming mode
        with open(video_path, 'rb') as video_file:
            # Create a multipart form
            files = {'video': (filename, video_file, 'video/avi')}
            
            # Send the POST request
            response = requests.post(
                api_url,
                auth=HTTPBasicAuth(username, password),
                headers=headers,
                data=payload,
                files=files,
                timeout=30.0  # Increased timeout for larger files
            )
            
            print(f"API Response Status: {response.status_code}")
            if response.status_code == 200:
                print("Video uploaded successfully")
                return True
            else:
                print(f"Upload failed with status: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error during video streaming: {e}")
        return False


import os
import glob
import time
from threading import Lock

def monitor_and_send_videos(video_directory, machine_id, machine_identifier, user_id):
    """Enhanced video monitoring with atomic lock file approach"""
    print(f"Starting video monitor thread for directory: {video_directory}")
    processed_videos = set()
    processing_lock = Lock()
    
    def create_lock_file(video_path):
        """Create atomic lock file - returns True if successful, False if already locked"""
        lock_path = video_path + '.processing'
        try:
            # Use O_CREAT | O_EXCL for atomic creation - fails if file exists
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{time.time()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except Exception as e:
            print(f"Error creating lock file: {e}")
            return False
    
    def remove_lock_file(video_path):
        """Remove lock file"""
        lock_path = video_path + '.processing'
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            print(f"Error removing lock file: {e}")
    
    def is_lock_stale(video_path, timeout=300):
        """Check if lock file is stale (older than timeout seconds)"""
        lock_path = video_path + '.processing'
        try:
            if not os.path.exists(lock_path):
                return False
            with open(lock_path, 'r') as f:
                timestamp = float(f.read().strip())
                return time.time() - timestamp > timeout
        except:
            return True  # Consider stale if we can't read it
    
    while True:
        try:
            video_pattern = os.path.join(video_directory, "*.avi")
            video_files = glob.glob(video_pattern)
            
            for video_path in video_files:
                with processing_lock:
                    # Skip if already processed
                    if video_path in processed_videos:
                        continue
                
                # Check for stale lock and clean it up
                if is_lock_stale(video_path):
                    print(f"Removing stale lock for: {video_path}")
                    remove_lock_file(video_path)
                
                # Try to acquire atomic lock
                if not create_lock_file(video_path):
                    # File is being processed by another thread/process
                    continue
                
                try:
                    print(f"Acquired lock for processing: {video_path}")
                    
                    # Check if file still exists (might have been deleted by another process)
                    if not os.path.exists(video_path):
                        with processing_lock:
                            processed_videos.add(video_path)
                        continue
                    
                    # Use the enhanced completion check
                    if is_file_complete_enhanced(video_path):
                        print(f"Found complete video file: {video_path}")
                        
                        # Double-check file still exists after completion check
                        if not os.path.exists(video_path):
                            print(f"File was deleted during processing: {video_path}")
                            with processing_lock:
                                processed_videos.add(video_path)
                            continue
                        
                        # Extract dataset info
                        filename = os.path.basename(video_path)
                        dataset_name = filename.replace('.avi', '')
                        
                        # Extract transaction_id
                        try:
                            parts = dataset_name.split('_')
                            transaction_id = parts[-1] if len(parts) > 2 else None
                        except:
                            transaction_id = None
                        
                        # Attempt upload
                        success = stream_video_to_api(
                            video_path, 
                            dataset_name, 
                            transaction_id, 
                            machine_id, 
                            user_id, 
                            machine_identifier
                        )
                        
                        if success:
                            print(f"Successfully uploaded: {video_path}")
                            try:
                                # Check if file exists before trying to delete
                                if os.path.exists(video_path):
                                    os.remove(video_path)
                                    print(f"Deleted uploaded video: {video_path}")
                                else:
                                    print(f"Video file already deleted: {video_path}")
                            except Exception as e:
                                print(f"Error deleting video file: {e}")
                        else:
                            os.remove(video_path)
                            print(f"Deleted uploaded video (Read timed out): {video_path}")
                            print(f"Failed to upload: {video_path}")
                        
                        # Mark as processed regardless of upload success
                        with processing_lock:
                            processed_videos.add(video_path)
                    else:
                        print(f"Video file not yet complete: {video_path}")
                        
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
                finally:
                    # Always remove lock file when done
                    remove_lock_file(video_path)
                    print(f"Released lock for: {video_path}")
            
            # Clean up processed videos set periodically to prevent memory growth
            with processing_lock:
                if len(processed_videos) > 100:
                    processed_videos = set(list(processed_videos)[-50:])
                    
        except Exception as e:
            print(f"Error in video monitoring thread: {e}")
        
        # Wait between checks to reduce CPU usage
        time.sleep(10)  # Check every 10 seconds

def is_file_complete_enhanced(file_path, stable_time=5):  # Reduced from 15 to 5 seconds
    """
    Enhanced file completion check specifically for video files
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        # Get initial file stats
        initial_stat = os.stat(file_path)
        initial_size = initial_stat.st_size
        initial_mtime = initial_stat.st_mtime
        
        # File must have some content
        if initial_size == 0:
            return False
        
        print(f"Checking completion for {file_path} (size: {initial_size} bytes)")
        
        # Wait for stability period (reduced time)
        time.sleep(stable_time)
        
        # Check again
        try:
            final_stat = os.stat(file_path)
            final_size = final_stat.st_size
            final_mtime = final_stat.st_mtime
        except OSError:
            # File might have been deleted or locked
            return False
        
        # Size and modification time should be unchanged
        if initial_size != final_size or initial_mtime != final_mtime:
            print(f"File still changing: size {initial_size}->{final_size}, mtime {initial_mtime}->{final_mtime}")
            return False
        
        # Try to open file exclusively to ensure it's not being written
        try:
            with open(file_path, 'r+b') as f:
                # Seek to end to verify file integrity
                f.seek(0, 2)  # Seek to end
                actual_size = f.tell()
                if actual_size != final_size:
                    return False
        except (IOError, OSError) as e:
            print(f"Cannot open file exclusively: {e}")
            return False
        
        print(f"File appears complete: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error in enhanced completion check: {e}")
        return False
    
class WebSocketDataManager:
    def __init__(self):
        self.current_data = {
            "validated_products": {
                "entry": {},
                "exit": {}
            },
            "invalidated_products": {
                "entry": {},
                "exit": {}
            }
        }
        print(f'sinii{self.current_data}')
        self._lock = threading.Lock()

    def update_data(self, new_data):
        with self._lock:
            self.current_data = new_data

    def get_current_data(self):
        with self._lock:
            return self.current_data.copy()
            
            
class TrackingData:
    def __init__(self):
        self.shutdown_event = Event()  # Add this line
        self.validated_products = {
            "entry": {},
            "exit": {}
        }
        self.invalidated_products = {
            "entry": {},
            "exit": {}
        }
        self.class_counters = {
            "entry": defaultdict(int),
            "exit": defaultdict(int)
        }
        self.counted_tracks = {
            "entry": set(),
            "exit": set()
        }
        self.machine_planogram = []
        self.machine_product_library = []
        self.hailo_pipeline_string = ""  # Add this line
        self.frame_rate_calc = 1
        self.last_time = time.time()
        self.websocket_data_manager = WebSocketDataManager()  # Add this line
        self.deposit = 0.0 
        self.machine_id = None
        self.machine_identifier = None
        self.user_id = None
        self.transaction_id = None
        
    def set_transaction_data(self, deposit, machine_id, machine_identifier, user_id, transaction_id):
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id


class HailoDetectionCallback(app_callback_class):
    def __init__(self, websocket=None,deposit = 0.0, machine_id=None, machine_identifier=None, user_id=None, transaction_id=None):
        super().__init__()
        self.tracking_data = TrackingData()
        
        self.use_frame = True
        self.websocket = websocket
        self.shutdown_event = Event() 
        self.deposit = deposit
        self.machine_id = machine_id
        self.machine_identifier = machine_identifier
        self.user_id = user_id
        self.transaction_id = transaction_id
        self.tracking_data.set_transaction_data(deposit, machine_id, machine_identifier, user_id, transaction_id)
        # Create video directory BEFORE loading planogram
        self.video_directory = os.path.join(os.getcwd(), "saved_videos")
        os.makedirs(self.video_directory, exist_ok=True)
        # Store machine_id persistently
        self.store_machine_id_env(machine_id)
        self.load_machine_planogram()
        self.load_machine_product_library()
        self.load_hailo_pipeline_config()  # Add this line
        
    # SOLUTION 2: Environment variable persistence
    def store_machine_id_env(self, machine_id):
        """Store machine_id as environment variable"""
        if machine_id is not None:
            os.environ['MACHINE_ID'] = str(machine_id)
            print(f"Machine ID {machine_id} stored in environment")
    
    def load_machine_id_env(self):
        """Load machine_id from environment variable"""
        return os.environ.get('MACHINE_ID')        
        
    def is_planogram_valid_for_machine(self, machine_id):
        """Check if current environment planogram is valid for the given machine ID"""
        try:
            # Check if we have a stored machine ID for this planogram
            stored_machine_id = os.environ.get('PLANOGRAM_MACHINE_ID')
            return stored_machine_id == str(machine_id) if stored_machine_id else False
        except Exception as e:
            print(f"Error checking planogram validity: {e}")
            return False
    
    def is_product_library_valid_for_machine(self, machine_id):
        """Check if current environment product library is valid for the given machine ID"""
        try:
            # Check if we have a stored machine ID for this product library
            stored_machine_id = os.environ.get('PRODUCT_LIBRARY_MACHINE_ID')
            return stored_machine_id == str(machine_id) if stored_machine_id else False
        except Exception as e:
            print(f"Error checking product library validity: {e}")
            return False

    def is_hailo_pipeline_valid_for_machine(self, machine_id):
        """Check if current environment hailo pipeline config is valid for the given machine ID"""
        try:
            # Check if we have a stored machine ID for this hailo pipeline config
            stored_machine_id = os.environ.get('HAILO_PIPELINE_MACHINE_ID')
            return stored_machine_id == str(machine_id) if stored_machine_id else False
        except Exception as e:
            print(f"Error checking hailo pipeline config validity: {e}")
            return False

    def store_planogram_env(self, planogram_data):
        """Store planogram data as environment variable with machine ID tracking"""
        try:
            # Convert planogram list to JSON string and store in environment
            planogram_json = json.dumps(planogram_data)
            os.environ['MACHINE_PLANOGRAM'] = planogram_json
            
            # Store the machine ID this planogram belongs to
            current_machine_id = self.load_machine_id_env()
            if current_machine_id:
                os.environ['PLANOGRAM_MACHINE_ID'] = str(current_machine_id)
            
            print(f"Planogram data stored in environment: {len(planogram_data)} products for machine {current_machine_id}")
            
            # Also update the tracking_data planogram
            self.tracking_data.machine_planogram = planogram_data
            
        except Exception as e:
            print(f"Error storing planogram in environment: {e}")
            
    def store_product_library_env(self, product_library_data):
        """Store product library data as environment variable with machine ID tracking"""
        try:
            # Convert product library list to JSON string and store in environment
            product_library_json = json.dumps(product_library_data)
            os.environ['MACHINE_PRODUCT_LIBRARY'] = product_library_json
            
            # Store the machine ID this product library belongs to
            current_machine_id = self.load_machine_id_env()
            if current_machine_id:
                os.environ['PRODUCT_LIBRARY_MACHINE_ID'] = str(current_machine_id)
            
            print(f"Product library data stored in environment: {len(product_library_data)} products for machine {current_machine_id}")
            
            # Also update the tracking_data product library
            self.tracking_data.machine_product_library = product_library_data
            
        except Exception as e:
            print(f"Error storing product library in environment: {e}")

    def store_hailo_pipeline_env(self, pipeline_data):
        """Store hailo pipeline string as environment variable with machine ID tracking"""
        try:
            # Convert complete pipeline data to JSON string and store in environment
            pipeline_json = json.dumps(pipeline_data)
        
            os.environ['HAILO_PIPELINE_DATA'] = pipeline_json
            
            # Store the machine ID this pipeline config belongs to
            current_machine_id = self.load_machine_id_env()
            if current_machine_id:
                os.environ['HAILO_PIPELINE_MACHINE_ID'] = str(current_machine_id)
            
            print(f"Hailo pipeline config stored in environment for machine {current_machine_id}")
            
            # Also update the tracking_data pipeline string
            if pipeline_data and len(pipeline_data) > 0:
                self.tracking_data.hailo_pipeline_string = pipeline_data[0].get('ai_config_string', '')
            
            
        except Exception as e:
            print(f"Error storing hailo pipeline config in environment: {e}")

    def load_planogram_env(self):
        """Load planogram data from environment variable"""
        try:
            planogram_json = os.environ.get('MACHINE_PLANOGRAM')
            if planogram_json:
                planogram_data = json.loads(planogram_json)
                #print(f"Planogram loaded from environment: {len(planogram_data)} products")
                return planogram_data
            else:
                print("No planogram found in environment")
                return []
        except Exception as e:
            print(f"Error loading planogram from environment: {e}")
            return []
            
    def load_product_library_env(self):
        """Load product library data from environment variable"""
        try:
            product_library_json = os.environ.get('MACHINE_PRODUCT_LIBRARY')
            if product_library_json:
                product_library_data = json.loads(product_library_json)
                #print(f"Product library loaded from environment: {len(product_library_data)} products")
                return product_library_data
            else:
                print("No product library found in environment")
                return []
        except Exception as e:
            print(f"Error loading product library from environment: {e}")
            return []

    def load_hailo_pipeline_env(self):
        """Load hailo pipeline data from environment variable"""
        try:
            pipeline_json = os.environ.get('HAILO_PIPELINE_DATA')
            if pipeline_json:
                pipeline_data = json.loads(pipeline_json)
                print("Hailo pipeline config loaded from environment")
                return pipeline_data
            else:
                print("No hailo pipeline config found in environment")
                return None
        except Exception as e:
            print(f"Error loading hailo pipeline config from environment: {e}")
            return None
    
    def load_machine_planogram(self):
        try:
            # Get machine_id from environment
            current_machine_id = self.load_machine_id_env()
            
            if not current_machine_id:
                print("No machine ID available - loading planogram from environment if available")
                # Try to load existing planogram from environment
                existing_planogram = self.load_planogram_env()
                if existing_planogram:
                    self.tracking_data.machine_planogram = existing_planogram
                    print(f"Loaded existing planogram from environment: {len(existing_planogram)} products")
                else:
                    self.tracking_data.machine_planogram = []
                    print("No planogram found in environment and no machine ID available")
                return

            # Check if planogram already exists in environment for this machine
            existing_planogram = self.load_planogram_env()
            if existing_planogram and self.is_planogram_valid_for_machine(current_machine_id):
                self.tracking_data.machine_planogram = existing_planogram
                print(f"Using existing planogram from environment for machine {current_machine_id}: {len(existing_planogram)} products")
                print("Skipping initial API fetch - valid planogram already exists in environment")
                
                # Only start the refresh thread, no initial API call
                self.start_planogram_refresh_thread()
                
                # Start video monitoring thread
                video_monitor_thread = threading.Thread(
                    target=monitor_and_send_videos,
                    args=(self.video_directory, current_machine_id, self.machine_identifier, self.user_id)
                )
                video_monitor_thread.daemon = True
                video_monitor_thread.start()
                print("Video monitoring thread started")
                
                return
            
            # If no valid planogram in environment for this machine, fetch from API
            print(f"No valid planogram found in environment for machine {current_machine_id} - fetching from API for initial setup")
            self.fetch_and_store_initial_planogram(current_machine_id)
            
        except Exception as e:
            print(f"Error loading planogram: {e}")
            # Final fallback - try to load from environment
            try:
                existing_planogram = self.load_planogram_env()
                if existing_planogram:
                    self.tracking_data.machine_planogram = existing_planogram
                    print("Using existing planogram from environment as final fallback")
                else:
                    self.tracking_data.machine_planogram = []
            except Exception as final_error:
                print(f"Final fallback error: {final_error}")
                self.tracking_data.machine_planogram = []
                
    def load_machine_product_library(self):
        try:
            # Get machine_id from environment
            current_machine_id = self.load_machine_id_env()
            
            if not current_machine_id:
                print("No machine ID available - loading product library from environment if available")
                # Try to load existing product library from environment
                existing_product_library = self.load_product_library_env()
                if existing_product_library:
                    self.tracking_data.machine_product_library = existing_product_library
                    print(f"Loaded existing product library from environment: {len(existing_product_library)} products")
                else:
                    self.tracking_data.machine_product_library = []
                    print("No product library found in environment and no machine ID available")
                return

            # Check if product library already exists in environment for this machine
            existing_product_library = self.load_product_library_env()
            if existing_product_library and self.is_product_library_valid_for_machine(current_machine_id):
                self.tracking_data.machine_product_library = existing_product_library
                print(f"Using existing product library from environment for machine {current_machine_id}: {len(existing_product_library)} products")
                print("Skipping initial API fetch - valid product library already exists in environment")
                
                # Only start the refresh thread, no initial API call
                self.start_product_library_refresh_thread()
                return
            
            # If no valid product library in environment for this machine, fetch from API
            print(f"No valid product library found in environment for machine {current_machine_id} - fetching from API for initial setup")
            self.fetch_and_store_initial_product_library(current_machine_id)
            
        except Exception as e:
            print(f"Error loading product library: {e}")
            # Final fallback - try to load from environment
            try:
                existing_product_library = self.load_product_library_env()
                if existing_product_library:
                    self.tracking_data.machine_product_library = existing_product_library
                    print("Using existing product library from environment as final fallback")
                else:
                    self.tracking_data.machine_product_library = []
            except Exception as final_error:
                print(f"Final fallback error: {final_error}")
                self.tracking_data.machine_product_library = []

    def load_hailo_pipeline_config(self):
        """Load hailo pipeline configuration similar to planogram and product library"""
        try:
            # Get machine_id from environment
            current_machine_id = self.load_machine_id_env()
            
            if not current_machine_id:
                print("No machine ID available - loading hailo pipeline config from environment if available")
                # Try to load existing pipeline config from environment
                existing_pipeline_config = self.load_hailo_pipeline_env()
                if existing_pipeline_config:
                    # Extract just ai_config_string for tracking_data
                    ai_config_string = existing_pipeline_config[0].get('ai_config_string', '') if existing_pipeline_config else ''
                    self.tracking_data.hailo_pipeline_string = ai_config_string
                    print("Loaded existing hailo pipeline config from environment")
                else:
                    # Use fallback pipeline string
                    self.tracking_data.hailo_pipeline_string = self.get_fallback_pipeline_string()
                    print("No hailo pipeline config found in environment and no machine ID available - using fallback")
                return

            # Check if pipeline config already exists in environment for this machine
            existing_pipeline_config = self.load_hailo_pipeline_env()
            if existing_pipeline_config and self.is_hailo_pipeline_valid_for_machine(current_machine_id):
                # Extract just ai_config_string for tracking_data
                ai_config_string = existing_pipeline_config[0].get('ai_config_string', '') if existing_pipeline_config else ''
                self.tracking_data.hailo_pipeline_string = ai_config_string
                print(f"Using existing hailo pipeline config from environment for machine {current_machine_id}")
                print("Skipping initial API fetch - valid hailo pipeline config already exists in environment")
                
                # Only start the refresh thread, no initial API call
                self.start_hailo_pipeline_refresh_thread()
                return
            
            # If no valid pipeline config in environment for this machine, fetch from API
            print(f"No valid hailo pipeline config found in environment for machine {current_machine_id} - fetching from API for initial setup")
            self.fetch_and_store_initial_hailo_pipeline_config(current_machine_id)
            
        except Exception as e:
            print(f"Error loading hailo pipeline config: {e}")
            # Final fallback - try to load from environment or use fallback
            try:
                existing_pipeline_config = self.load_hailo_pipeline_env()
                if existing_pipeline_config:
                    ai_config_string = existing_pipeline_config[0].get('ai_config_string', '') if existing_pipeline_config else ''
                    self.tracking_data.hailo_pipeline_string = ai_config_string
                    print("Using existing hailo pipeline config from environment as final fallback")
                else:
                    self.tracking_data.hailo_pipeline_string = self.get_fallback_pipeline_string()
                    print("Using hardcoded fallback pipeline string")
            except Exception as final_error:
                print(f"Final fallback error: {final_error}")
                self.tracking_data.hailo_pipeline_string = self.get_fallback_pipeline_string()

    def get_fallback_pipeline_string(self):
        """Return the fallback pipeline string when API fetch fails"""
        return "hailoroundrobin mode=0 name=fun ! queue name=hailo_pre_infer_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! hailonet hef-path=resources/realWorld7.hef batch-size=2 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! queue name=hailo_postprocess0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! hailofilter function-name=filter_letterbox so-path=/home/afiq/hailo-rpi5-examples/basic_pipelines/../resources/libyolo_hailortpp_postprocess.so config-path=resources/labelsBaru.json qos=false ! queue name=hailo_track0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 keep-new-frames=1 keep-tracked-frames=1 keep-lost-frames=1 keep-past-metadata=true ! hailostreamrouter name=sid src_0::input-streams=\"<sink_0>\" src_1::input-streams=\"<sink_1>\"  compositor name=comp start-time-selection=0 sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=350 sink_1::ypos=0 ! queue name=hailo_video_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! videoconvert ! queue name=hailo_display_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! fpsdisplaysink video-sink=autovideosink name=hailo_display sync=false text-overlay=true v4l2src device=/dev/video0 name=source_0 ! video/x-raw, width=640, height=360 ! queue name=source_scale_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! videoscale name=source_videoscale_0 n-threads=2 ! queue name=source_convert_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=3 name=source_convert_0 qos=false ! video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! queue name=inference_wrapper_input_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! fun.sink_0  sid.src_0 ! queue name=identity_callback_q_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! identity name=identity_callback_0 ! queue name=hailo_draw_0 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! hailooverlay ! videoscale n-threads=8 ! video/x-raw,width=640,height=360 ! queue name=comp_q_0 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! comp.sink_0 v4l2src device=/dev/video2 name=source_2 ! video/x-raw, width=640, height=360 ! queue name=source_scale_q_2 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! videoscale name=source_videoscale_2 n-threads=2 ! queue name=source_convert_q_2 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=3 name=source_convert_2 qos=false ! video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! queue name=inference_wrapper_input_q_2 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! fun.sink_1  sid.src_1 ! queue name=identity_callback_q_1 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! identity name=identity_callback_1 ! queue name=hailo_draw_1 leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! hailooverlay ! videoscale n-threads=8 ! video/x-raw,width=640,height=360 ! queue name=comp_q_1 leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! comp.sink_1"

    def fetch_and_store_initial_planogram(self, machine_id):
        """Fetch planogram from API only for initial setup when not in environment"""
        try:
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            api_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/machine/Machine_listing/machine_planogram/{machine_id}'
            
            # Start video monitoring thread
            video_monitor_thread = threading.Thread(
                target=monitor_and_send_videos,
                args=(self.video_directory, machine_id, self.machine_identifier, self.user_id)
            )
            video_monitor_thread.daemon = True
            video_monitor_thread.start()
            print("Video monitoring thread started")

            # Start refresh thread for future updates
            self.start_planogram_refresh_thread()

            # Initial API fetch (only when planogram doesn't exist in environment)
            api_response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password), headers=headers)
            
            if api_response.status_code == 200:
                machine_planogram = api_response.json().get('machine_planogram', [])
                
                # Store in environment and update tracking_data
                self.store_planogram_env(machine_planogram)
                
                print("Initial planogram fetched and stored in environment:")
                for product in machine_planogram:
                    print(f"Product library ID: {product['product_library_id']}, Name: {product['product_name']}")
                    
            else:
                print(f"Initial API request failed: {api_response.status_code}")
                self.tracking_data.machine_planogram = []
                
        except Exception as e:
            print(f"Error in initial API request: {e}")
            self.tracking_data.machine_planogram = []
            
    def fetch_and_store_initial_product_library(self, machine_id):
        """Fetch product library from API only for initial setup when not in environment"""
        try:
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            api_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/fetch_productlibrary/{machine_id}'

            # Start refresh thread for future updates
            self.start_product_library_refresh_thread()

            # Initial API fetch (only when product library doesn't exist in environment)
            api_response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password), headers=headers)
            
            if api_response.status_code == 200:
                machine_product_library = api_response.json().get('data', [])
                
                # Store in environment and update tracking_data
                self.store_product_library_env(machine_product_library)
                
                print("Initial product library fetched and stored in environment:")
                for product in machine_product_library:
                    print(f"Product ID: {product['product_id']}, Name: {product['product_name']}")
                    
            else:
                print(f"Initial product library API request failed: {api_response.status_code}")
                self.tracking_data.machine_product_library = []
                
        except Exception as e:
            print(f"Error in initial product library API request: {e}")
            self.tracking_data.machine_product_library = []

    def fetch_and_store_initial_hailo_pipeline_config(self, machine_id):
        """Fetch hailo pipeline config from API only for initial setup when not in environment"""
        try:
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key,'machine_id': machine_id}
            
            api_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/aiconfig/AiConfig/get_selected_config_bymachine'

            # Start refresh thread for future updates
            self.start_hailo_pipeline_refresh_thread()

            # Initial API fetch (only when pipeline config doesn't exist in environment)
            api_response = requests.get(api_endpoint, auth=HTTPBasicAuth(username, password), headers=headers)
            
            if api_response.status_code == 200:
                response_data = api_response.json()
                hailo_pipeline_data = response_data.get('data', [])
                
                if hailo_pipeline_data:
                    # Store in environment and update tracking_data
                    self.store_hailo_pipeline_env(hailo_pipeline_data)
                    print("Initial hailo pipeline config fetched and stored in environment")
                else:
                    print("API response doesn't contain hailo_pipeline_string - using fallback")
                    fallback_string = self.get_fallback_pipeline_string()
                    self.store_hailo_pipeline_env(fallback_string)
                    
            else:
                print(f"Initial hailo pipeline config API request failed: {api_response.status_code} - using fallback")
                fallback_string = self.get_fallback_pipeline_string()
                fallback_data = [{'ai_config_string': fallback_string}]
                self.store_hailo_pipeline_env(fallback_string)
                
        except Exception as e:
            print(f"Error in initial hailo pipeline config API request: {e} - using fallback")
            fallback_string = self.get_fallback_pipeline_string()
            fallback_data = [{'ai_config_string': fallback_string}]
            self.store_hailo_pipeline_env(fallback_string)

    def start_planogram_refresh_thread(self):
        """Start the background refresh thread for planogram updates"""
        def refresh_planogram():
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            while True:
                try:
                    # Always get the latest machine_id
                    refresh_machine_id = self.load_machine_id_env()
                    if not refresh_machine_id:
                        print("No machine ID available for refresh - skipping")
                        time.sleep(1000)
                        continue
                        
                    refresh_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/machine/Machine_listing/machine_planogram/{refresh_machine_id}'
                    
                    api_response = requests.get(refresh_endpoint, 
                                             auth=HTTPBasicAuth(username, password), 
                                             headers=headers)
                    
                    if api_response.status_code == 200:
                        new_planogram = api_response.json().get('machine_planogram', [])
                        
                        # Check if planogram has changed before updating
                        current_planogram = self.load_planogram_env()
                        if new_planogram != current_planogram:
                            # Store the updated planogram in environment
                            self.store_planogram_env(new_planogram)
                            print(f"Planogram updated in environment: {len(new_planogram)} products")
                        else:
                            print("Planogram unchanged - no update needed")
                            
                    else:
                        print(f"API refresh failed: {api_response.status_code}")
                        
                except Exception as e:
                    print(f"Error refreshing planogram: {e}")
                
                time.sleep(1000)  # Refresh every 1000 seconds

        # Start refresh thread
        refresh_thread = threading.Thread(target=refresh_planogram)
        refresh_thread.daemon = True
        refresh_thread.start()
        print("Planogram refresh thread started")
        
    def start_product_library_refresh_thread(self):
        """Start the background refresh thread for product library updates"""
        def refresh_product_library():
            username = 'admin'
            password = '1234'
            api_key = '123456'
            headers = {'x-api-key': api_key}
            
            while True:
                try:
                    # Always get the latest machine_id
                    refresh_machine_id = self.load_machine_id_env()
                    if not refresh_machine_id:
                        print("No machine ID available for product library refresh - skipping")
                        time.sleep(1000)
                        continue
                        
                    refresh_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/fetch_productlibrary/{refresh_machine_id}'
                    
                    api_response = requests.get(refresh_endpoint, 
                                             auth=HTTPBasicAuth(username, password), 
                                             headers=headers)
                    
                    if api_response.status_code == 200:
                        new_product_library = api_response.json().get('data', [])
                        
                        # Check if product library has changed before updating
                        current_product_library = self.load_product_library_env()
                        if new_product_library != current_product_library:
                            # Store the updated product library in environment
                            self.store_product_library_env(new_product_library)
                            print(f"Product library updated in environment: {len(new_product_library)} products")
                        else:
                            print("Product library unchanged - no update needed")
                            
                    else:
                        print(f"Product library API refresh failed: {api_response.status_code}")
                        
                except Exception as e:
                    print(f"Error refreshing product library: {e}")
                
                time.sleep(1000)  # Refresh every 1000 seconds

        # Start refresh thread
        refresh_thread = threading.Thread(target=refresh_product_library)
        refresh_thread.daemon = True
        refresh_thread.start()
        print("Product library refresh thread started")

    def start_hailo_pipeline_refresh_thread(self):
        """Start the background refresh thread for hailo pipeline config updates"""
        def refresh_hailo_pipeline_config():
            username = 'admin'
            password = '1234'
            api_key = '123456'
            refresh_machine_id = self.load_machine_id_env()
            headers = {'x-api-key': api_key,'machine_id': refresh_machine_id}
            
            while True:
                try:
                    # Always get the latest machine_id
                    #refresh_machine_id = self.load_machine_id_env()
                    if not refresh_machine_id:
                        print("No machine ID available for hailo pipeline config refresh - skipping")
                        time.sleep(1000)
                        continue
                        
                    refresh_endpoint = f'https://stg-sfapi.nuboxtech.com/index.php/mobile_app/aiconfig/AiConfig/get_selected_config_bymachine'
                    
                    api_response = requests.get(refresh_endpoint, 
                                             auth=HTTPBasicAuth(username, password), 
                                             headers=headers)
                    
                    if api_response.status_code == 200:
                        response_data = api_response.json()
                        new_pipeline_data = response_data.get('data', '')
                        
                        if new_pipeline_data:
                            # Check if pipeline config has changed before updating
                            current_pipeline_data = self.load_hailo_pipeline_env()
                            if new_pipeline_data != current_pipeline_data:
                                # Store the updated pipeline config in environment
                                self.store_hailo_pipeline_env(new_pipeline_data)
                                print("Hailo pipeline config updated in environment")
                            else:
                                print("Hailo pipeline config unchanged - no update needed")
                        else:
                            print("API response doesn't contain hailo_pipeline_data")
                            
                    else:
                        print(f"Hailo pipeline config API refresh failed: {api_response.status_code}")
                        
                except Exception as e:
                    print(f"Error refreshing hailo pipeline config: {e}")
                
                time.sleep(1000)  # Refresh every 1000 seconds

        # Start refresh thread
        refresh_thread = threading.Thread(target=refresh_hailo_pipeline_config)
        refresh_thread.daemon = True
        refresh_thread.start()
        print("Hailo pipeline config refresh thread started")

    def get_planogram_from_env(self):
        """Get current planogram from environment (useful for external access)"""
        return self.load_planogram_env()
        
    def get_product_library_from_env(self):
        """Get current product library from environment (useful for external access)"""
        return self.load_product_library_env()
        
    def get_hailo_pipeline_from_env(self):
        """Get current hailo pipeline config from environment (useful for external access)"""
        return self.load_hailo_pipeline_env()

    def get_ai_config_string_from_env(self):
        """Extract just the ai_config_string from the stored pipeline data"""
        try:
            pipeline_data = self.load_hailo_pipeline_env()
            if pipeline_data and len(pipeline_data) > 0:
                return pipeline_data[0].get('ai_config_string', '')
            else:
                return ''
        except Exception as e:
            print(f"Error extracting ai_config_string: {e}")
            return ''
        

    def validate_detected_product(self, detected_product):
        # Get the current product library from environment to ensure we have the latest data
        current_product_library = self.load_product_library_env()
        if current_product_library:
            self.tracking_data.machine_product_library = current_product_library
        
        # Get the current planogram from environment to ensure we have the latest data
        current_planogram = self.load_planogram_env()
        if current_planogram:
            self.tracking_data.machine_planogram = current_planogram
            
        # Normalize the detected product for comparison
        normalized_detected_product = detected_product.replace(' ', '').lower()
    

        # Find the detected product in the product library
        detected_product_details = None
        for product in self.tracking_data.machine_product_library:
            product_name = product.get('product_name', '')
            normalized_product_name = product_name.replace(' ', '').lower()
            if normalized_product_name == normalized_detected_product:
                detected_product_details = product
                break
        
        if not detected_product_details:
            return {
                "valid": False,
                "product_details": None,
                "message": f"{detected_product} not found in product library"
            }
        
        # Get the product_id from the detected product in product library
        detected_product_id = detected_product_details.get('product_id')
        if not detected_product_id:
            return {
                "valid": False,
                "product_details": detected_product_details,
                "message": f"{detected_product} found in library but missing product_id"
            }
        
        # Check if this product_id exists in the machine planogram
        # Note: planogram uses 'product_library_id' which should match 'product_id' from product library
        matching_planogram_products = [
            product for product in self.tracking_data.machine_planogram 
            if str(product.get('product_library_id', '')) == str(detected_product_id)
        ]
        
        if matching_planogram_products:
            return {
                "valid": True,
                "product_details": matching_planogram_products[0],
                "message": f"{detected_product} validated successfully - found in both library and planogram"
            }
        else:
            return {
                "valid": False,
                "product_details": None,
                "message": f"{detected_product} found in library but not available in machine planogram"
            }
            
        
            
            
class HailoDetectionApp:
    def __init__(self, app_callback, user_data):
        self.app_callback = app_callback
        self.user_data = user_data
        self.door_monitor_active = True
        self.door_monitor_thread = threading.Thread(target=self.monitor_door)
        self.door_monitor_thread.daemon = True
        
        self.use_frame = True
        self.labels_json = 'resources/labelsBaru.json'
        self.hef_path = 'resources/yolov6n.hef'
        self.arch = 'hailo8'
        self.show_fps = True
        
        # Set up Hailo pipeline configuration
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        
        # Post-processing configuration
        self.post_process_so = os.path.join(os.path.dirname(__file__), '../resources/libyolo_hailortpp_postprocess.so')
        self.post_function_name = "filter_letterbox"
        self.create_pipeline()
        self.door_monitor_thread.start()

    def get_pipeline_string(self):
        """Get pipeline string from the loaded hailo pipeline configuration"""
        # Get the current hailo pipeline string from environment
        pipeline_string = self.user_data.get_ai_config_string_from_env()
        
        # If no pipeline string is available, try to load from environment
        if not pipeline_string:
            pipeline_string = self.user_data.tracking_data.hailo_pipeline_string
        
        # If still no pipeline string, use fallback
        if not pipeline_string:
            print("No hailo pipeline config found - using fallback pipeline string")
            pipeline_string = self.user_data.get_fallback_pipeline_string()
        
        print(f'pipeline here: {pipeline_string}')
        return pipeline_string

    def create_pipeline(self):
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        self.pipeline = Gst.parse_launch(pipeline_string)
        self.loop = GLib.MainLoop()

        # Set up bus call
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        
        # Connect callbacks for both identity elements
        for stream_id in [0, 1]:
            identity = self.pipeline.get_by_name(f"identity_callback_{stream_id}")
            if identity:
                pad = identity.get_static_pad("src")
                if pad:
                    # Store probe IDs for later removal
                    callback_data = {"user_data": self.user_data, "stream_id": stream_id}
                    probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, callback_data)
                    if not hasattr(self, 'probe_ids'):
                        self.probe_ids = {}
                    self.probe_ids[f"stream_{stream_id}"] = (identity, pad, probe_id)
                    print(f"Successfully added probe to identity element for stream {stream_id}")
                else:
                    print(f"Warning: Could not get src pad from identity element for stream {stream_id}")
            else:
                print(f"Warning: Could not find identity_callback_{stream_id} element in pipeline")
    
        return True
    # Rest of the methods remain the same
    def monitor_door(self):
        start_time = time.time()
        """Monitor door switch and trigger shutdown when door closes"""
        while self.door_monitor_active :
            door_sw = GPIO.input(DOOR_SWITCH_PIN)
            if door_sw == 0 and time.time() - start_time > 5:  # Door is closed
                print("Door closed - Initiating shutdown")
                self.shutdown()
                break
            time.sleep(0.1)
    
    def shutdown(self, signum=None, frame=None):
        print("Shutting down... Please wait.")
        
        # Stop door monitoring
        self.door_monitor_active = False
        # Set shutdown events
        self.user_data.tracking_data.shutdown_event.set()
        self.user_data.shutdown_event.set()
        
        # Stop pipeline
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay
        
        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay
        
        self.pipeline.set_state(Gst.State.NULL)
        
        # Force close any remaining windows
        cv2.destroyAllWindows()
        
        # Quit the main loop
        GLib.idle_add(self.loop.quit)
        
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True

    def run(self):
        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)
        
        
        
        if self.use_frame:
            display_process = multiprocessing.Process(
                target=display_user_data_frame, 
                args=(self.user_data,)
            )
            display_process.start()
        
        try:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop.run()
        except Exception as e:
            print(f"Error in pipeline: {e}")
        finally:
            self.user_data.tracking_data.shutdown_event.set()
            self.user_data.shutdown_event.set()
            
            self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()

from scipy.spatial import distance


# Global variables for cross-camera tracking
global_track_counter = 0
local_to_global_id_map = {}  # Maps (camera_id, local_track_id) to global_track_id
global_movement_history = defaultdict(deque)  # Movement history for global IDs
global_last_counted_direction = {}  # Last counted direction for global IDs
global_track_labels = {}  # Store labels for each global ID

# New structure to track active objects per camera and label
active_objects_per_camera = {
    0: defaultdict(dict),  # camera_id -> label -> {local_track_id: global_track_id}
    1: defaultdict(dict)
}

# Cross-camera matching candidates
cross_camera_candidates = defaultdict(list)  # label -> [list of (camera_id, local_track_id, global_track_id)]

# Adjust the movement_history and last_counted_direction to be per-camera
camera_movement_history = {
    0: defaultdict(lambda: deque(maxlen=5)),
    1: defaultdict(lambda: deque(maxlen=5))
}

def get_global_track_id(camera_id, local_track_id, features=None, label=None):
    """
    Get or create a global track ID for the given local track ID and camera.
    
    Logic:
    1. Same local_track_id on same camera always gets same global_id
    2. Different local_track_ids with same label on same camera get different global_ids
    3. Objects with same label on different cameras get same global_id (cross-camera matching)
    4. Multiple objects with same label on different cameras get different global_ids
    
    Args:
        camera_id: ID of the camera
        local_track_id: Local track ID assigned by the tracker
        features: Feature vector (ignored for now)
        label: Class label of the detected object
    Returns:
        Global track ID
    """
    global global_track_counter, local_to_global_id_map, global_track_labels
    global active_objects_per_camera, cross_camera_candidates
    
    # Check if this local track already has a global ID
    if (camera_id, local_track_id) in local_to_global_id_map:
        return local_to_global_id_map[(camera_id, local_track_id)]
    
    if not label:
        # If no label, just create a new global ID
        new_global_id = global_track_counter
        global_track_counter += 1
        local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
        return new_global_id
    
    # Check for cross-camera matching opportunity
    other_camera = 1 if camera_id == 0 else 0
    
    # Look for objects with the same label on the other camera that aren't matched yet
    available_matches = []
    if label in active_objects_per_camera[other_camera]:
        for other_local_id, other_global_id in active_objects_per_camera[other_camera][label].items():
            # Check if this global_id is already being used by another object on our camera
            already_matched_on_this_camera = False
            for our_local_id, our_global_id in active_objects_per_camera[camera_id][label].items():
                if our_global_id == other_global_id:
                    already_matched_on_this_camera = True
                    break
            
            if not already_matched_on_this_camera:
                available_matches.append((other_local_id, other_global_id))
    
    # If we found an available match on the other camera, use its global ID
    if available_matches:
        # Use the first available match (you could implement more sophisticated matching here)
        matched_local_id, matched_global_id = available_matches[0]
        local_to_global_id_map[(camera_id, local_track_id)] = matched_global_id
        active_objects_per_camera[camera_id][label][local_track_id] = matched_global_id
        global_track_labels[matched_global_id] = label
        return matched_global_id
    
    # No cross-camera match found, create a new global ID
    new_global_id = global_track_counter
    global_track_counter += 1
    local_to_global_id_map[(camera_id, local_track_id)] = new_global_id
    active_objects_per_camera[camera_id][label][local_track_id] = new_global_id
    global_track_labels[new_global_id] = label
    
    return new_global_id

def cleanup_inactive_tracks(camera_id, active_local_track_ids):
    """
    Clean up tracking data for tracks that are no longer active
    
    Args:
        camera_id: ID of the camera
        active_local_track_ids: Set of currently active local track IDs
    """
    global local_to_global_id_map, active_objects_per_camera
    
    # Find inactive tracks for this camera
    inactive_tracks = []
    for (cam_id, local_id), global_id in local_to_global_id_map.items():
        if cam_id == camera_id and local_id not in active_local_track_ids:
            inactive_tracks.append((cam_id, local_id, global_id))
    
    # Remove inactive tracks
    for cam_id, local_id, global_id in inactive_tracks:
        # Remove from local_to_global_id_map
        del local_to_global_id_map[(cam_id, local_id)]
        
        # Remove from active_objects_per_camera
        label = global_track_labels.get(global_id)
        if label and label in active_objects_per_camera[cam_id]:
            if local_id in active_objects_per_camera[cam_id][label]:
                del active_objects_per_camera[cam_id][label][local_id]
                
                # Clean up empty label entries
                if not active_objects_per_camera[cam_id][label]:
                    del active_objects_per_camera[cam_id][label]

def analyze_movement_direction(track_id, center, tracking_data, camera_id, global_id):
    """
    Analyze movement direction based on 5 consecutive frames using global IDs
    Args:
        track_id: The local ID of the tracked object
        center: Current center point (x, y)
        tracking_data: TrackingData instance containing counted_tracks
        camera_id: ID of the camera (0 or 1)
        global_id: Global ID for this track
    Returns: 'entry' for upward movement, 'exit' for downward movement, None for undefined
    """
    # Store movement in camera-specific history
    camera_movement_history[camera_id][track_id].appendleft(center)
    
    # Copy to global movement history (to maintain analysis consistency)
    global_movement_history[global_id].appendleft((center, camera_id))
    
    # Wait until we have enough frames to analyze for this camera
    if len(camera_movement_history[camera_id][track_id]) < 5:
        return None
        
    # Calculate vertical movement
    first_y = camera_movement_history[camera_id][track_id][-1][1]  # Oldest position
    last_y = camera_movement_history[camera_id][track_id][0][1]    # Newest position
    
    # Calculate average movement between consecutive points
    total_movement = 0
    for i in range(1, len(camera_movement_history[camera_id][track_id])):
        curr_y = camera_movement_history[camera_id][track_id][i-1][1]
        prev_y = camera_movement_history[camera_id][track_id][i][1]
        total_movement += curr_y - prev_y
    
    avg_movement = total_movement / 4  # We have 4 intervals between 5 points
    
    # Use a threshold to determine significant movement (e.g., 5 pixels)
    threshold = 5
    
    if abs(avg_movement) < threshold:
        return None
    
    # Determine current direction
    current_direction = 'exit' if avg_movement > 0 else 'entry'
    
    # Check if direction has changed since last count for this global ID
    if global_id in global_last_counted_direction:
        if current_direction != global_last_counted_direction[global_id]:
            # Direction changed, remove global_id from the old direction's counted set
            if global_last_counted_direction[global_id] in tracking_data.counted_tracks:
                tracking_data.counted_tracks[global_last_counted_direction[global_id]].discard(global_id)
    
    # Update global movement direction and last counted direction
    global_last_counted_direction[global_id] = current_direction
        
    return current_direction

def detection_callback(pad, info, callback_data):
    global camera_covered, cover_alert_thread, blink
    user_data = callback_data["user_data"]
    stream_id = callback_data["stream_id"]
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    format, width, height = get_caps_from_pad(pad)
    if not all([format, width, height]):
        return Gst.PadProbeReturn.OK

    # Process detections first
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Get video frame for visualization
    frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Check if frame is dark/covered
    if is_frame_dark(frame):
        camera_covered = True
        if cover_alert_thread is None or not cover_alert_thread.is_alive():
            cover_alert_thread = threading.Thread(target=handle_cover_alert, daemon=True)
            cover_alert_thread.start()
    else:
        camera_covered = False
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Ensure buzzer is off when camera is not covered

    # Collect active track IDs for cleanup
    active_local_track_ids = set()
    
    # Process detections and draw on frame
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        class_id = detection.get_class_id()
        
        # Get track ID from Hailo tracker
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            active_local_track_ids.add(track_id)
        
        # Calculate bounding box coordinates
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Get or create global track ID
        global_id = get_global_track_id(stream_id, track_id, None, label)
        
        validation_result = user_data.validate_detected_product(label)
        color = compute_color_for_labels(class_id)
        
        # Draw bounding box and label with track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} L:{track_id} G:{global_id} {'Valid' if validation_result['valid'] else 'Invalid'}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if validation_result['valid'] else (0, 0, 255), 2)
        
        # Draw trail
        draw_trail(frame, track_id, center, color, global_id=global_id)
        
        # Analyze movement direction and update counters
        direction = analyze_movement_direction(track_id, center, user_data.tracking_data, stream_id, global_id)
        
        if direction:
            # Only increment counter if we haven't counted this global_id's movement yet
            # or if the direction has changed since last count
            if (global_id not in user_data.tracking_data.counted_tracks.get(direction, set()) or
                (global_id in global_last_counted_direction and 
                 direction != global_last_counted_direction[global_id])):
                     
                user_data.tracking_data.class_counters[direction][label] += 1
                if direction not in user_data.tracking_data.counted_tracks:
                    user_data.tracking_data.counted_tracks[direction] = set()
                user_data.tracking_data.counted_tracks[direction].add(global_id)
                
                # Update validated/invalidated products
                if validation_result['valid']:
                    if label not in user_data.tracking_data.validated_products[direction]:
                        user_data.tracking_data.validated_products[direction][label] = {
                            "count": 0,
                            "product_details": validation_result['product_details']
                        }
                    user_data.tracking_data.validated_products[direction][label]["count"] += 1
                else:
                    if label not in user_data.tracking_data.invalidated_products[direction]:
                        user_data.tracking_data.invalidated_products[direction][label] = {
                            "count": 0,
                            "raw_detection": {
                                "name": label,
                                "confidence": confidence,
                                "tracking_id": global_id,
                                "bounding_box": {
                                    "xmin": x1,
                                    "ymin": y1,
                                    "xmax": x2,
                                    "ymax": y2
                                }
                            }
                        }
                    user_data.tracking_data.invalidated_products[direction][label]["count"] += 1

    # Clean up inactive tracks for this camera
    cleanup_inactive_tracks(stream_id, active_local_track_ids)

    # Draw FPS and other overlays
    current_time = time.time()
    user_data.tracking_data.last_time = current_time
    
    # Draw counts
    label = next((det.get_label() for det in detections), None)
    draw_counts(frame, user_data.tracking_data.class_counters, label)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Store frames in user_data
    if stream_id == 0:
        user_data.frame_left = frame
    elif stream_id == 1:
        user_data.frame_right = frame
        
    # Check if both frames are available
    if hasattr(user_data, "frame_left") and hasattr(user_data, "frame_right"):
        combined_frame = np.hstack((user_data.frame_left, user_data.frame_right))
        user_data.set_frame(combined_frame)
    
    # Update websocket data
    websocket_data = {
        "validated_products": {
            "entry": {
                product: {
                    "count": details["count"],
                    "product_details": details["product_details"]
                } for product, details in user_data.tracking_data.validated_products["entry"].items()
            },
            "exit": {
                product: {
                    "count": details["count"],
                    "product_details": details["product_details"]
                } for product, details in user_data.tracking_data.validated_products["exit"].items()
            }
        },
        "invalidated_products": {
            "entry": {
                product: {
                    "count": details["count"],
                    "raw_detection": details["raw_detection"]
                } for product, details in user_data.tracking_data.invalidated_products["entry"].items()
            },
            "exit": {
                product: {
                    "count": details["count"],
                    "raw_detection": details["raw_detection"]
                } for product, details in user_data.tracking_data.invalidated_products["exit"].items()
            }
        }
    }
    
    user_data.tracking_data.websocket_data_manager.update_data(websocket_data)
    
    # Calculate prices and control buzzer
    current_data = user_data.tracking_data.websocket_data_manager.get_current_data()
    deposit = user_data.deposit
    total_price = calculate_total_price_and_control_buzzer(current_data, deposit)
    
    return Gst.PadProbeReturn.OK
                
                
async def run_tracking(websocket: WebSocket):
    global readyToProcess,cover_alert_thread
    global unlock_data, done
    unlock_data = 0
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None
    product_name = None
    image_count = None
    #time.sleep(0.5)
    #GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
    # Wait for start_preview message
    while True:
        try:
            message_text = await websocket.receive_text()
            print(f"Ada pa woii: {message_text}")
            try:
                message = json.loads(message_text)
                if isinstance(message, dict) and message.get('action') == 'start_preview':
                    unlock_data = 1
                    deposit = float(message.get('deposit', 0.0))
                    machine_id = message.get('machine_id')
                    machine_identifier = message.get('machine_identifier')
                    user_id = message.get('user_id')
                    transaction_id = message.get('transaction_id')
                    product_name = message.get('product_name')
                    image_count = message.get('image_count')
                    
                    
                    print(f"Deposit: {deposit}")
                    print(f"Machine ID: {machine_id}")
                    print(f"Machine Identifier: {machine_identifier}")
                    print(f"User ID: {user_id}")
                    print(f"Transaction ID: {transaction_id}")
                    print(f"Product Name: {product_name}")
                    print(f"Image Count: {image_count}")
                    
                    print(f"Start preview received. Unlock data: {unlock_data}")
                   
                    break
                else:
                    break
            except json.JSONDecodeError:
                continue
        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": f"Error waiting for start message: {str(e)}"
            })
            return

    # Handle door control
    if unlock_data == 1:   
        print("Unlock door for 0.5 seconds")
        #GPIO.output(DOOR_LOCK_PIN, GPIO.LOW)
        readyToProcess= True
        unlock_data = 0
        print("Unlock data reset to 0")


    
    
    try:
        if isinstance(message, dict) and message.get('action') == 'product_upload':
                    
            done = True        
            machine_id = message.get('machine_id')
            machine_identifier = message.get('machine_identifier')
            user_id = message.get('user_id')
            product_name = message.get('product_name')
            image_count = message.get('image_count')
                
                   
            print(f"Machine ID: {machine_id}")
            print(f"Machine Identifier: {machine_identifier}")
            print(f"User ID: {user_id}")
            print(f"Product Name: {product_name}")
            print(f"Image Count: {image_count}")
            
            print("\n" + "="*50)
            print("STARTING IMAGE CAPTURE PROCESS")
            print("="*50)
            print(f"Total images to capture: {image_count * 2} ({image_count} per camera)")
            
            
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(3)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            
            time.sleep(2)
            
            # Capture from camera1 (/dev/video0) FIRST
            print("\n" + "-"*30)
            print("CAMERA 1 CAPTURE PHASE")
            print("-"*30)
            camera1_images = capture_images(0, image_count)  # or capture_images_interactive(0, image_count)
            
            
            # Small break between cameras
            print("\n" + "="*20)
            print("SWITCHING TO CAMERA 2")
            print("="*20)
            
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(3)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            
            time.sleep(2)
        
            # Capture from camera2 (/dev/video2) SECOND
            print("\n" + "-"*30)
            print("CAMERA 2 CAPTURE PHASE")
            print("-"*30)
            camera2_images = capture_images(2, image_count)  # or capture_images_interactive(2, image_count)
        
            if camera1_images and camera2_images:
                print("\nAll images captured successfully!")
            
                # Upload images to API
                print("\nUploading images to API...")
                if upload_images_to_api(camera1_images, camera2_images,machine_id,machine_identifier, user_id, product_name, image_count):
                    print("Images uploaded successfully!")
                    
                    # Delete all captured images after successful upload
                    print("\nDeleting captured images...")
                    all_images = camera1_images + camera2_images
                    delete_images(all_images)
                    
                    # Sound buzzer to indicate success
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    time.sleep(3)
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                else:
                    print("Failed to upload images.")
                    print("Images will be kept for retry or manual inspection.") 
                    
            else:
                print("Failed to capture all images.")      
                   
        else:
            # Initialize door status monitoring
            door_monitor_active = True
            done = True
            async def monitor_door():
                nonlocal door_monitor_active
                while door_monitor_active:
                    door_sw = 1
                    if door_sw == 0:  # Door is closed
                        print("Door closed - Stopping tracking")
                        callback.tracking_data.shutdown_event.set()
                        callback.shutdown_event.set()
                        door_monitor_active = False
                        # Send final message to client
                        try:
                           await websocket.send_json({
                             "status": "stopped",
                             "message": "Door closed - Tracking stopped"
                           })
                        except Exception as e:
                           print(f"Error sending final message: {e}")
                        break
                    await asyncio.sleep(0.1)
            # Start door monitoring task
            door_monitor_task = asyncio.create_task(monitor_door())
    
            # Initialize Hailo detection
            callback = HailoDetectionCallback(websocket,deposit,machine_id, machine_identifier, user_id, transaction_id)
    
            def send_websocket_data():
                while not callback.tracking_data.shutdown_event.is_set():
                    try:
                        current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                        # Get deposit from user_data (you'll need to add this to your data structure)
                
    
                        # Calculate prices and control buzzer
                        #total_price = calculate_total_price_and_control_buzzer(current_data, deposit)
                        #print(f"SINI WEY: {current_data}")
                        asyncio.run(websocket.send_json(current_data))
                        time.sleep(1)  # Send updates every second
                    except Exception as e:
                        print(f"Error sending websocket data: {e}")

            # Start websocket data sender thread
            websocket_sender = threading.Thread(target=send_websocket_data)
            websocket_sender.start()
    
    
            def signal_handler(signum, frame):
                 print("\nCtrl+C detected. Initiating shutdown...")
                 callback.tracking_data.shutdown_event.set()
                 callback.shutdown_event.set()
                 # Force close any remaining windows
                 cv2.destroyAllWindows()
            
            # Set up signal handler
            signal.signal(signal.SIGINT, signal_handler)

    

            app = HailoDetectionApp(detection_callback, callback)            
            app.run()
    except Exception as e:
        print(f"Error during tracking: {e}")
    finally:
        # Ensure cleanup happens
            await websocket.send_json({
                "status": "stopped",
                "message": "Tracking has been fullyyy stopped"
            })
            door_monitor_active = False
            if cover_alert_thread is not None and cover_alert_thread.is_alive():
              camera_covered = False
              GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Ensure buzzer is off
              cover_alert_thread.join()
              cover_alert_thread = None
              alert_thread.join()
              alert_thread = None
            await door_monitor_task
            callback.tracking_data.shutdown_event.set()
            callback.shutdown_event.set()
            websocket_sender.join()
            app.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            




def capture_images(device_id, num_images=3):
    """Optimized camera capture with faster settings."""
    image_paths = []
    try:
        # Open the camera
        cap = cv2.VideoCapture(device_id)
        
        # OPTIMIZATION 1: Use MJPEG format (much faster than YUYV)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # OPTIMIZATION 2: Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # OPTIMIZATION 3: Increase FPS and reduce buffer
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # OPTIMIZATION 4: Disable auto-exposure for consistent timing (optional)
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {device_id}")
            return []
            
        print(f"\n=== Starting capture for Camera {device_id} ===")
        print("Get ready to show your products!")
        
        
        
        # Capture images
        for i in range(1, num_images + 1):
            print(f"\nCamera {device_id}: Product position {i} now!")
            
            
            time.sleep(0.5)  
            
            
            for _ in range(5):  
                cap.read()
            
            print(f"Capturing image {i}...")
            
            # Capture the actual frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture image {i} from camera {device_id}")
                continue
            
            # Save the image
            filename = os.path.join('camera_images', f"camera_{device_id}_image_{i}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"Saved {filename}")
            image_paths.append(filename)
            
            
            try:
                
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                time.sleep(1)  # Shorter buzzer time
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
            except:
                pass  # Skip if no GPIO
            
            # Reduced wait time between captures
            if i < num_images:
                print(f"Get ready for product position {i+1}...")
                time.sleep(1)  # Keep this for user positioning
                
        print(f"\n=== Finished capturing from Camera {device_id} ===")
        cap.release()
        return image_paths
        
    except Exception as e:
        print(f"Error with camera {device_id}: {e}")
        if 'cap' in locals():
            cap.release()
        return []

def upload_images_to_api(camera1_images, camera2_images, machine_id, machine_identifier, user_id, product_name, image_count):
    """Upload images to the API."""
    api_url = "https://stg-sfapi.nuboxtech.com/index.php/mobile_app/product/Product/upload_product_images"
    
    # Authentication
    username = 'admin'
    password = '1234'
    api_key = '123456'
    
    payload = {
        'machine_id': machine_id,
        'machine_identifier': machine_identifier,
        'user_id': user_id,
        'product_name': product_name,
        'image_count': image_count
    }
    
    # Prepare headers
    headers = {'x-api-key': api_key}
    
    files = []
    opened_files = []  # Keep track of opened file handles
    
    try:
        # Add Fantech camera images
        for i, img_path in enumerate(camera1_images):
            file_handle = open(img_path, 'rb')
            opened_files.append(file_handle)
            files.append(('image[]', (f'camera1{i}.jpg', file_handle, 'image/jpeg')))
        
        # Add USB camera images
        for i, img_path in enumerate(camera2_images):
            file_handle = open(img_path, 'rb')
            opened_files.append(file_handle)
            files.append(('image[]', (f'camera2{i}.jpg', file_handle, 'image/jpeg')))
        
        # Upload to API
        response = requests.post(
            api_url,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            data=payload,
            files=files,
        )
        
        print("API Response Status Code:", response.status_code)
        print("API Response:", response.text)
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error uploading to API: {e}")
        return False
    finally:
        # Always close all opened file handles
        for file_handle in opened_files:
            try:
                file_handle.close()
            except:
                pass     

def delete_images(image_paths):
    """Delete image files from the filesystem."""
    deleted_count = 0
    for img_path in image_paths:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted: {img_path}")
                deleted_count += 1
            else:
                print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Error deleting {img_path}: {e}")
    
    print(f"Successfully deleted {deleted_count} images")
    return deleted_count        



import threading
import time
import subprocess
import os
import tempfile
from threading import Lock
from gtts import gTTS
import pygame
import io
from pathlib import Path

class TTSManager:
    def __init__(self):
        self.tts_lock = Lock()
        self.audio_lock = Lock()  # Separate lock for general audio playback
        self.init_audio_player()
    
    def init_audio_player(self):
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Audio player initialized successfully")
        except Exception as e:
            print(f"Error initializing audio player: {e}")
    
    def play_mp3(self, file_path, volume=0.7, wait_for_completion=True):
        """
        Play an MP3 file
        
        Args:
            file_path (str): Path to the MP3 file
            volume (float): Volume level (0.0 to 1.0)
            wait_for_completion (bool): Whether to wait for playback to complete
        
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        def _play():
            with self.audio_lock:
                try:
                    # Check if file exists
                    if not os.path.exists(file_path):
                        print(f"MP3 file not found: {file_path}")
                        return False
                    
                    # Check if file is MP3
                    if not file_path.lower().endswith('.mp3'):
                        print(f"File is not an MP3: {file_path}")
                        return False
                    
                    print(f"Playing MP3: {file_path}")
                    
                    # Load and play the MP3 file
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play()
                    
                    if wait_for_completion:
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    
                    print(f"Finished playing MP3: {file_path}")
                    return True
                    
                except Exception as e:
                    print(f"Error playing MP3 {file_path}: {e}")
                    return False
        
        if wait_for_completion:
            # Run synchronously
            return _play()
        else:
            # Run asynchronously
            audio_thread = threading.Thread(target=_play, daemon=True)
            audio_thread.start()
            return True
    
    def play_mp3_async(self, file_path, volume=0.7):
        """Play MP3 file asynchronously (non-blocking)"""
        return self.play_mp3(file_path, volume, wait_for_completion=False)
    
    def play_mp3_sync(self, file_path, volume=0.7):
        """Play MP3 file synchronously (blocking)"""
        return self.play_mp3(file_path, volume, wait_for_completion=True)
    
    def play_sound_effect(self, file_path, volume=0.7):
        """
        Play a sound effect using pygame.mixer.Sound (for short audio clips)
        This allows multiple sounds to play simultaneously
        """
        try:
            if not os.path.exists(file_path):
                print(f"Sound file not found: {file_path}")
                return False
            
            print(f"Playing sound effect: {file_path}")
            sound = pygame.mixer.Sound(file_path)
            sound.set_volume(volume)
            sound.play()
            return True
            
        except Exception as e:
            print(f"Error playing sound effect {file_path}: {e}")
            return False
    
    def stop_all_audio(self):
        """Stop all audio playback"""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()  # Stop all sound effects
            print("All audio stopped")
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def pause_audio(self):
        """Pause current music playback"""
        try:
            pygame.mixer.music.pause()
            print("Audio paused")
        except Exception as e:
            print(f"Error pausing audio: {e}")
    
    def resume_audio(self):
        """Resume paused music playback"""
        try:
            pygame.mixer.music.unpause()
            print("Audio resumed")
        except Exception as e:
            print(f"Error resuming audio: {e}")
    
    def set_volume(self, volume):
        """Set the volume for music playback (0.0 to 1.0)"""
        try:
            pygame.mixer.music.set_volume(volume)
            print(f"Volume set to: {volume}")
        except Exception as e:
            print(f"Error setting volume: {e}")
    
    def is_audio_playing(self):
        """Check if audio is currently playing"""
        try:
            return pygame.mixer.music.get_busy()
        except:
            return False
    
    def get_audio_position(self):
        """Get current position in music playback (if supported)"""
        try:
            return pygame.mixer.music.get_pos()
        except:
            return -1
    
    # ... [Keep all existing TTS methods] ...
    
    def speak_async(self, text, lang='en'):
        """Speak text asynchronously using gTTS with improved error handling"""
        def _speak():
            with self.tts_lock:
                try:
                    print(f"Speaking: {text}")
                    
                    # Create gTTS object
                    tts = gTTS(text=text, lang=lang, slow=False)
                    
                    # Save to BytesIO buffer instead of file
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    
                    # Load and play audio using pygame
                    pygame.mixer.music.load(audio_buffer)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    print(f"Finished speaking: {text}")
                    
                except Exception as e:
                    print(f"Error in gTTS: {e}")
                    self.fallback_speak(text)
        
        # Run TTS in separate thread
        tts_thread = threading.Thread(target=_speak, daemon=True)
        tts_thread.start()
    
    def speak_with_file(self, text, lang='ms'):
        """Alternative method using temporary file (more reliable for some systems)"""
        def _speak():
            with self.tts_lock:
                try:
                    print(f"Speaking: {text}")
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                        temp_filename = temp_file.name
                    
                    # Generate speech and save to file
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(temp_filename)
                    
                    # Play using pygame
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                    
                    print(f"Finished speaking: {text}")
                    
                except Exception as e:
                    print(f"Error in gTTS with file: {e}")
                    self.fallback_speak(text)
        
        # Run TTS in separate thread
        tts_thread = threading.Thread(target=_speak, daemon=True)
        tts_thread.start()
    
    def fallback_speak(self, text):
        """Fallback TTS using system espeak"""
        try:
            # Check if espeak is available
            subprocess.run(['which', 'espeak'], check=True, capture_output=True)
            
            # Use espeak with slower rate and better settings
            cmd = [
                'espeak', 
                '-s', '120',    # Speech rate (words per minute)
                '-a', '200',    # Higher amplitude
                '-p', '50',     # Pitch
                '-g', '3',      # Gap between words
                text
            ]
            subprocess.run(cmd, check=False, capture_output=True)
            print(f"Fallback TTS spoke: {text}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Espeak not available. Install with: sudo apt-get install espeak")
            self.alternative_fallback(text)
    
    def alternative_fallback(self, text):
        """Alternative fallback using festival if available"""
        try:
            subprocess.run(['which', 'festival'], check=True, capture_output=True)
            
            # Create temporary file for festival
            temp_file = '/tmp/tts_temp.txt'
            with open(temp_file, 'w') as f:
                f.write(text)
            
            cmd = ['festival', '--tts', temp_file]
            subprocess.run(cmd, check=False, capture_output=True)
            
            # Clean up
            os.remove(temp_file)
            print(f"Festival TTS spoke: {text}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Festival not available. Install with: sudo apt-get install festival")
            print(f"No TTS available. Would speak: {text}")
    
    def speak_door_open(self):
        """Speak door open message in English"""
        self.speak_async("Open the door", lang='en')
    
    def speak_door_close(self):
        """Speak door close message in English"""
        self.speak_async("Door is closing", lang='en')
    
    def speak_english(self, text):
        """Speak text in English"""
        self.speak_async(text, lang='en')
    
    def speak_malay(self, text):
        """Speak text in Malay"""
        self.speak_async(text, lang='ms')
    
    def test_voice(self):
        """Test the TTS voice with sample phrases"""
        print("Testing English voice...")
        self.speak_english("Testing voice clarity. Can you hear this clearly?")
        time.sleep(3)
        print("Testing Malay voice...")
        self.speak_malay("Ujian suara yang jelas. Boleh dengar dengan baik?")
    
    def test_mp3_playback(self):
        """Test MP3 playback functionality"""
        print("Testing MP3 playback...")
        # You would need to have test MP3 files for this
        test_files = [
            "test_sound.mp3",
            "welcome.mp3",
            "notification.mp3"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"Testing: {file_path}")
                self.play_mp3_sync(file_path, volume=0.5)
                time.sleep(1)
            else:
                print(f"Test file not found: {file_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_all_audio()
            pygame.mixer.quit()
        except:
            pass

# Create global TTS manager instance
tts_manager = TTSManager()

async def cleanup_websocket_sounds():
    """Clean up all sounds when WebSocket connection closes"""
    global camera_covered_sound_playing, price_alert_sound_playing
    
    camera_covered_sound_playing = False
    price_alert_sound_playing = False
    tts_manager.stop_all_audio()
    print("WebSocket cleanup - all sounds stopped")

def play_camera_alert_sound():
    """Play camera covered alert sound"""
    tts_manager.play_mp3_async("sounds/siren1.mp3", volume=1.0)

def play_price_alert_sound():
    """Play price exceeded alert sound"""
    tts_manager.play_mp3_async("sounds/siren1.mp3", volume=1.0)

def stop_all_alert_sounds():
    """Stop all alert sounds"""
    global camera_covered_sound_playing, price_alert_sound_playing
    camera_covered_sound_playing = False
    price_alert_sound_playing = False
    tts_manager.stop_all_audio()
    
# Example usage functions for your WebSocket endpoint
def play_welcome_sound():
    """Play welcome sound when door opens"""
    tts_manager.play_mp3_async("sounds/welcome.mp3", volume=0.8)

def play_goodbye_sound():
    """Play goodbye sound when door closes"""
    tts_manager.play_mp3_async("sounds/goodbye.mp3", volume=0.8)

def play_notification_sound():
    """Play notification sound"""
    tts_manager.play_sound_effect("sounds/notification.mp3", volume=0.6)

def play_error_sound():
    """Play error sound"""
    tts_manager.play_sound_effect("sounds/error.mp3", volume=0.7)

# Test the voice when module loads (optional)
# tts_manager.test_voice()

done = False
@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    global readyToProcess, done
    deposit = 0.0
    machine_id = None
    machine_identifier = None
    user_id = None
    transaction_id = None
    websocket_sender = None
    
    await websocket.accept()
    
    # Voice announcement: Door opened
    print("WebSocket connected - announcing door open")
    tts_manager.speak_door_open()
    GPIO.output(DOOR_LOCK_PIN, GPIO.LOW)
    GPIO.output(LED_RED, GPIO.LOW)
    GPIO.output(LED_GREEN, GPIO.LOW)
    try:
       start_time = time.time()
       readyToProcess = True		
       while readyToProcess and time.time() - start_time < 5:
         door_sw = 1
         if door_sw == 1:
          await run_tracking(websocket)
          readyToProcess = False
          
         else:
           readyToProcess = True
       
       if not done:

           callback = HailoDetectionCallback(websocket, deposit, machine_id, machine_identifier, user_id, transaction_id)
           while not callback.tracking_data.shutdown_event.is_set():
                try:
                    current_data = callback.tracking_data.websocket_data_manager.get_current_data()
                    await websocket.send_json(current_data)
                    await asyncio.sleep(1)  # Send updates every second
                    break
                except Exception as e:
                    print(f"Error sending websocket data: {e}")
                    break   
    except Exception as e:
        print(f"WebSocket tracking error: {e}")
    finally:
        # Clean up sounds before closing
        await cleanup_websocket_sounds()
        # Voice announcement: Door closing
        print("WebSocket closing - announcing door close")
        tts_manager.speak_door_close()
        GPIO.output(DOOR_LOCK_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED_GREEN, GPIO.HIGH)
        GPIO.output(LED_RED, GPIO.HIGH)
        await websocket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    
    
    args = parser.parse_args()
    
    atexit.register(GPIO.cleanup)
    
    uvicorn.run(
        "app_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )


if __name__ == "__main__":
    main()
