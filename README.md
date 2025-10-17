# License Plate Recognition - Real-time Detection

This project implements real-time license plate detection using YOLOv8 and optional OCR text recognition.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Model Options:
   - **Custom trained model**: Place your trained `best.pt` in the project directory
   - **Pre-trained fallback**: Use `--use-pretrained` flag to use YOLOv8 COCO model (less accurate)
   - **Any .pt file**: Script will automatically detect any .pt files in the directory

## Usage

### Test on validation dataset:
```bash
python detect_camera.py --source valid
```

### Basic camera detection:
```bash
python detect_camera.py
```

### With pre-trained model (if no custom weights):
```bash
python detect_camera.py --use-pretrained
```

### With custom parameters:
```bash
python detect_camera.py --weights ./best.pt --source 0 --conf 0.5 --save-crops --ocr
```

### Parameters:
- `--weights`: Path to YOLOv8 model weights (default: ./best.pt)
- `--source`: Camera index, video file path, or 'valid' for validation dataset
- `--conf`: Confidence threshold (default: 0.25)
- `--save-crops`: Save cropped license plate detections
- `--save-dir`: Directory to save crops (default: ./crops)
- `--imgsz`: Inference image size (default: 640)
- `--ocr`: Enable OCR text recognition
- `--use-pretrained`: Use pre-trained YOLOv8 model if custom weights not found

### Controls during detection:
**Camera/Video mode:**
- `q`: Quit the application
- `s`: Save current frame
- `c`: Toggle crop saving on/off

**Validation dataset mode:**
- `q`: Quit the application
- `s`: Save current frame
- `n` or `Space`: Next image

## Troubleshooting

### "Model weights not found" Error:
1. **Train your model**: Use YOLOv8 to train on your dataset and save as `best.pt`
2. **Use pre-trained model**: Add `--use-pretrained` flag
3. **Download weights**: Place any YOLOv8 .pt file in the project directory
4. **Specify path**: Use `--weights path/to/your/model.pt`

### Validation Dataset:
The script looks for validation images in these locations:
- `valid/images/`
- `../valid/images/`
- `./valid/images/`
- `data/valid/images/`

Make sure your validation images are in one of these directories.

## Features

- Real-time license plate detection from webcam
- Validation dataset testing mode
- Automatic model fallback options
- Color-coded confidence levels (Green: High, Yellow: Medium, Orange: Low)
- Optional OCR text recognition using EasyOCR
- Live FPS counter and detection statistics
- Automatic crop saving with metadata
- Interactive controls during runtime

## Model Information

The model expects license plates as defined in the data.yaml configuration:
- Classes: 1 (License_Plate)
- Input size: 640x640 (configurable)

For best results, train a custom YOLOv8 model on license plate data.
