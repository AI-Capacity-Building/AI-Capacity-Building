# Object Detection and Recognition using YOLO

## Prerequisites

<ul>
<li>Python 3</li>
<li>OpenCV 4</li>
<li>Numpy</li>
<li>YOLOv3 pre-trained models:</li>
  <ul>
  <li>YOLOv3 config: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg</li>
  <li>YOLOv3 weights: https://pjreddie.com/media/files/yolov3.weights</li>
  <li>Class names: https://github.com/pjreddie/darknet/blob/master/data/coco.names</li>
  </ul>
</ul>

## Usage

Copy the above config, weights and class names files into the same folder as this source code.

To detect object in image, just run:

```Python
python yolo_detect_image.py --image name_of_your_image_here
```

For example, with this input image:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/YOLO-example.png?raw=true" alt="YOLO input image" title="YOLO input image">
<br>

The output will be:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/YOLO-output.png?raw=true" alt="YOLO input image" title="YOLO input image">
<br>

Similarly, to detect object in video, just run:

```python
python yolo_detect_video.py --video name_of_your_video_here
```

An video example can be seen below:

[![](http://img.youtube.com/vi/5Zt7ohK2Rjk/0.jpg)](http://www.youtube.com/watch?v=5Zt7ohK2Rjk "")

Please feel free to adjust CONF_THRESHOLD and NMS_THRESHOLD constants to suit your needs.

Full tutorial is available at http://dangminhthang.com/computer-vision/object-detection-and-recognition-using-yolo/
