import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True, help='Image source: file, folder, video, "usb0", or "picamera0"')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold (e.g., 0.4)')
parser.add_argument('--resolution', default=None, help='Resolution in WxH (e.g., "640x480")')
parser.add_argument('--record', action='store_true', help='Record video or webcam output as "demo1.avi"')
parser.add_argument('--save_image', action='store_true', help='Save processed images with bounding boxes')
args = parser.parse_args()

# -------------------------------
# Setup
# -------------------------------
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
save_image = args.save_image

# Ensure model exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or not found.')
    sys.exit(1)

# Load YOLO model
model = YOLO(model_path)
labels = model.names

# Create output folder for saved images
output_folder = "deployed_images"
os.makedirs(output_folder, exist_ok=True)

# Supported file extensions
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# Determine source type
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid source: {img_source}')
    sys.exit(1)

# Parse resolution
resize = False
if user_res:
    resW, resH = map(int, user_res.lower().split('x'))
    resize = True

# Setup recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording is only supported for video and camera inputs.')
        sys.exit(1)
    if not user_res:
        print('Please specify resolution using --resolution to enable recording.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load input source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Bounding box colors (Tableau 10)
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]

# -------------------------------
# Inference Loop
# -------------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

while True:
    t_start = time.perf_counter()

    # Get frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('End of stream or camera error. Exiting.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Error reading from Picamera. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(det.cls.item())
        conf = det.conf.item()
        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            xmin, ymin, xmax, ymax = xyxy
            classname = labels[classidx]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # FPS display
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Object count display
    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show results
    cv2.namedWindow('YOLO detection results', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO detection results', frame)

    # Save to video
    if record:
        recorder.write(frame)

    # Save image if requested
    if save_image:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"detected_image_{timestamp}.jpg" if source_type in ['image', 'folder'] else f"detected_video_frame_{timestamp}.jpg"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved image: {output_path}")

    # Key handling
    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite("capture.png", frame)

    # FPS calculation
    t_stop = time.perf_counter()
    frame_rate = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -------------------------------
# Cleanup
# -------------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
