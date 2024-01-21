import cv2
import numpy as np
import argparse
import mlflow

from utils import process_frame

# Define constants
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Path to input video', required=True)
args = parser.parse_args()

# Read COCO dataset classes
with open('object_detection_example/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNet("object_detection_example/yolov3.weights", "object_detection_example/yolov3.cfg", "darknet")

# Get the output layer names used for the forward pass
outNames = net.getUnconnectedOutLayersNames()

# Create MLflow run
with mlflow.start_run():

    # Log video path as an artifact
    mlflow.log_artifact(args.video)

    writer = None

    cap = cv2.VideoCapture(args.video)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            # Break the loop if the video ends
            break

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the input
        net.setInput(blob)

        # Run forward pass
        outs = net.forward(outNames)

        # Process output and draw predictions
        process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)

        # Save video
        if writer is None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            writer = cv2.VideoWriter('out2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frameWidth, frameHeight))

        writer.write(frame)

        # Display the frame and wait for a key press
        cv2.imshow("YOLO Object Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Log progress every 10 frames
        if frame_count % 10 == 0:
            mlflow.log_metric("frame_processed", frame_count)

        frame_count += 1

    # Log the total number of frames processed
    mlflow.log_metric("total_frames_processed", frame_count)

# cleaning up
cap.release()
writer.release()
cv2.destroyAllWindows()
