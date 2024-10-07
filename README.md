
---

# YOLOv5 Object Detection with Live Browser Image Capture

This project demonstrates how to use YOLOv5 for real-time object detection on images captured from your browser. It is designed to run on Google Colab and uses the Ultralytics YOLOv5 model along with OpenCV for image processing.

## Features
- Capture images directly from your browser using Google Colab.
- Use a pretrained YOLOv5x model for object detection.
- Display real-time detection results.

## Setup Instructions

### Step 1: Install Required Libraries

Start by installing the necessary libraries for PyTorch, OpenCV, and YOLOv5 dependencies:
```bash
!pip install torch torchvision torchaudio
!pip install opencv-python
```

### Step 2: Clone the YOLOv5 Repository

Next, clone the YOLOv5 repository from GitHub:
```bash
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
```

### Step 3: Capturing an Image in Colab

This project uses a custom JavaScript function to capture an image directly from your browser. The image is saved locally and then processed by the YOLOv5 model for object detection. The `take_photo` function handles this process.

```python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import torch
from PIL import Image

def take_photo(filename='photo.jpg', quality=0.99):
    js = Javascript('''
    async function takePhoto(quality) {
        const div = document.createElement('div');
        const video = document.createElement('video');
        const button = document.createElement('button');
        button.textContent = 'Capture';
        div.appendChild(video);
        div.appendChild(button);
        document.body.appendChild(div);

        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await video.play();

        video.style.width = '50%';
        await new Promise((resolve) => button.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getTracks().forEach(track => track.stop());
        div.remove();

        return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Capture photo
photo_path = take_photo()
```

### Step 4: Running Object Detection

Once the image is captured, YOLOv5x (pretrained) will perform object detection on the image.

```python
# Load YOLOv5x model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Read and preprocess the captured image
img = cv2.imread(photo_path)

# Run inference
results = model(img)

# Display the results
results.show()
```

### Example Output

- Once the script runs, you will be able to capture a photo from your browser, and the YOLOv5 model will display the detected objects within the image.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Google Colab for image capture and real-time inference

## Notes
- This project is designed to run in Google Colab, where the `take_photo` function is used to interact with the browser for image capture.
- The YOLOv5 model (`yolov5x`) is used for detecting objects in the captured image.

## References
- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [PyTorch Hub Documentation](https://pytorch.org/hub/)

---
