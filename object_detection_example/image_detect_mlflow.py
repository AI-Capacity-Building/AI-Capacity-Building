import cv2
import numpy as np
import mlflow
import mlflow.pyfunc
import os

# Define constants
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
weights_path = "object_detection_example/yolov3.weights"
config_path = "object_detection_example/yolov3.cfg"
coco_names_path = "object_detection_example/coco.names"
image_path = "object_detection_example/YOLOimage.jpg"

# Create MLflow run
with mlflow.start_run():

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, config_path)

    classes = []
    with open(coco_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(layer_names)

    # Get bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any indices were returned
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Log image with bounding boxes to MLflow
    mlflow.log_artifact(image_path)
    
    # Log model parameters
    mlflow.log_param("CONF_THRESHOLD", CONF_THRESHOLD)
    mlflow.log_param("NMS_THRESHOLD", NMS_THRESHOLD)
    mlflow.log_param("weights_path", weights_path)
    mlflow.log_param("config_path", config_path)
    mlflow.log_param("coco_names_path", coco_names_path)

    # Log metrics (you can add more metrics as needed)
    mlflow.log_metric("num_detected_objects", len(indices))

    # Save the resulting image
    output_image_path = "output_image.jpg"
    cv2.imshow("YOLO Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_image_path, image)

    mlflow.log_artifact(output_image_path)

# Log additional metrics or artifacts as needed

# Start MLflow UI for visualization
    os.system("mlflow ui")
