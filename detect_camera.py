"""
Real-time license plate detection from webcam using YOLOv8 (ultralytics).
Usage:
  python detect_camera.py --weights ./best.pt --source 0 --conf 0.25 --save-crops
"""
import argparse
import time
import os
import cv2
import numpy as np
import glob

# Try to import OCR library for text recognition
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not installed. Install with 'pip install easyocr' for text recognition.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./best.pt", help="path to model weights (YOLOv8 .pt)")
    parser.add_argument("--source", default=0, help="camera index, video file path, or 'valid' for validation dataset")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--save-crops", action="store_true", help="save cropped detections")
    parser.add_argument("--save-dir", type=str, default="./crops", help="directory to save crops")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--ocr", action="store_true", help="enable OCR text recognition")
    parser.add_argument("--use-pretrained", action="store_true", help="use pre-trained YOLOv8 model if custom weights not found")
    return parser.parse_args()

def load_model(weights, use_pretrained=False):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package not found. Install via `pip install ultralytics`") from e

    # Check if custom weights exist
    if os.path.exists(weights):
        print(f"Loading custom model: {weights}")
        model = YOLO(weights)
        return model

    # If custom weights don't exist, try to find any .pt files in directory
    pt_files = glob.glob("*.pt")
    if pt_files:
        weights_file = pt_files[0]
        print(f"Custom weights not found. Using: {weights_file}")
        model = YOLO(weights_file)
        return model

    # If no custom weights found and use_pretrained is True, use pre-trained model
    if use_pretrained:
        print("No custom weights found. Using pre-trained YOLOv8n model.")
        print("Note: This model is trained on COCO dataset and may not detect license plates accurately.")
        print("For better results, train a custom model or use --use-pretrained flag.")
        model = YOLO('yolov8n.pt')  # This will download if not present
        return model

    # If we get here, no model could be loaded
    raise FileNotFoundError(f"""
Model weights not found: {weights}

Options to fix this:
1. Train a model and save weights as 'best.pt'
2. Download pre-trained weights and place in current directory
3. Use --use-pretrained flag to use YOLOv8 COCO model (less accurate for license plates)
4. Specify different weights path with --weights argument

Available .pt files in directory: {pt_files if pt_files else 'None'}
""")

def initialize_ocr():
    """Initialize OCR reader if available"""
    if OCR_AVAILABLE:
        try:
            reader = easyocr.Reader(['en'])
            return reader
        except Exception as e:
            print(f"OCR initialization failed: {e}")
            return None
    return None

def extract_text_from_crop(reader, crop_img):
    """Extract text from license plate crop using OCR"""
    if reader is None:
        return ""

    try:
        # Preprocess image for better OCR
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        results = reader.readtext(thresh, detail=0)
        if results:
            # Join all detected text and clean it
            text = ' '.join(results).strip()
            # Filter out non-alphanumeric characters except spaces and dashes
            text = ''.join(c for c in text if c.isalnum() or c in ' -')
            return text
    except Exception as e:
        print(f"OCR error: {e}")

    return ""

def draw_boxes(frame, boxes, scores, classes, names, texts=None, color=(0,255,0)):
    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(boxes.astype(int), scores, classes)):
        # Use different colors based on confidence
        if conf > 0.7:
            box_color = (0, 255, 0)  # Green for high confidence
        elif conf > 0.5:
            box_color = (0, 255, 255)  # Yellow for medium confidence
        else:
            box_color = (0, 165, 255)  # Orange for low confidence

        label = f"{names.get(int(cls), 'License_Plate')} {conf:.2f}"

        # Add OCR text if available
        if texts and i < len(texts) and texts[i]:
            label += f" | {texts[i]}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw label background
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        y_text = max(y1, t_size[1] + 4)
        cv2.rectangle(frame, (x1, y_text - t_size[1] - 4), (x1 + t_size[0] + 6, y_text + 2), box_color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

def get_validation_images():
    """Get list of validation images from data.yaml configuration"""
    # Read from data.yaml first
    try:
        import yaml
        if os.path.exists('data.yaml'):
            with open('data.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
                val_path = data_config.get('val', '')
                if val_path and os.path.exists(val_path):
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                    images = []
                    for ext in image_extensions:
                        images.extend(glob.glob(os.path.join(val_path, ext)))
                    if images:
                        print(f"Found {len(images)} validation images in: {val_path}")
                        return images
    except ImportError:
        pass

    # Fallback to manual search
    valid_paths = [
        "valid/images",
        "../valid/images",
        "./valid/images",
        "data/valid/images",
        "train/images"  # Add as fallback
    ]

    for path in valid_paths:
        if os.path.exists(path):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            images = []
            for ext in image_extensions:
                images.extend(glob.glob(os.path.join(path, ext)))
            if images:
                print(f"Found {len(images)} validation images in: {path}")
                return images

    print("No validation images found. Checked paths:", valid_paths)
    return []

def run_camera(args):
    model = load_model(args.weights, args.use_pretrained)
    names = getattr(model, "names", {0: 'License_Plate'}) or {0: 'License_Plate'}

    # Initialize OCR if requested
    ocr_reader = None
    if args.ocr:
        ocr_reader = initialize_ocr()
        if ocr_reader:
            print("OCR enabled - license plate text will be extracted")
        else:
            print("OCR initialization failed - continuing without text recognition")

    # Handle validation dataset
    if args.source == 'valid':
        validation_images = get_validation_images()
        if not validation_images:
            raise RuntimeError("No validation images found. Please check your dataset structure.")

        print(f"Running detection on {len(validation_images)} validation images")
        print("Controls: 'q' - Quit, 'n' - Next image, 's' - Save current frame")

        os.makedirs(args.save_dir, exist_ok=True)

        for img_idx, img_path in enumerate(validation_images):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Could not load image: {img_path}")
                continue

            print(f"Processing image {img_idx + 1}/{len(validation_images)}: {os.path.basename(img_path)}")

            start = time.time()

            # Run inference
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            res = results[0]

            # Extract detection data
            boxes = np.empty((0,4))
            scores = np.array([])
            classes = np.array([])
            texts = []

            if hasattr(res, "boxes") and len(res.boxes) > 0:
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                except Exception:
                    boxes = np.array(res.boxes.xyxy)
                    scores = np.array(res.boxes.conf)
                    classes = np.array(res.boxes.cls)

                # Extract text from license plates if OCR is enabled
                if ocr_reader and boxes.shape[0] > 0:
                    for (x1, y1, x2, y2) in boxes.astype(int):
                        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            text = extract_text_from_crop(ocr_reader, crop)
                            texts.append(text)
                        else:
                            texts.append("")

            # Draw detections
            draw_boxes(frame, boxes, scores, classes, names, texts if ocr_reader else None)

            # Add image info
            info_text = [
                f"Image: {img_idx + 1}/{len(validation_images)}",
                f"File: {os.path.basename(img_path)}",
                f"Detections: {len(boxes)}",
                f"Confidence: {args.conf}",
            ]

            if ocr_reader:
                info_text.append("OCR: ON")

            for i, text in enumerate(info_text):
                y_pos = 30 + i * 25
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Save crops if enabled
            if args.save_crops and boxes.shape[0] > 0:
                for i, ((x1,y1,x2,y2), conf, cls) in enumerate(zip(boxes.astype(int), scores, classes)):
                    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    text_suffix = f"_{texts[i].replace(' ', '_')}" if texts and i < len(texts) and texts[i] else ""
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    fname = os.path.join(args.save_dir, f"{base_name}_det{i}_cls{int(cls)}_{conf:.2f}{text_suffix}.jpg")
                    cv2.imwrite(fname, crop)

            # Display frame
            cv2.imshow("License Plate Detection - Validation Dataset", frame)

            # Handle key presses
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save current frame
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(args.save_dir, f"{base_name}_detected.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Frame saved: {save_path}")
            elif key == ord("n") or key == 32:  # 'n' or spacebar for next
                continue

        cv2.destroyAllWindows()
        return

    # Original camera/video logic
    try:
        source = int(args.source)  # Try to convert to int for camera
    except ValueError:
        source = args.source  # Use as string for video file

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/source: {args.source}")

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    os.makedirs(args.save_dir, exist_ok=True)
    frame_id = 0
    detection_count = 0

    print("License Plate Detection Started")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'c' - Toggle crop saving")

    save_crops_toggle = args.save_crops

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame_id += 1
        start = time.time()

        # Run inference
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        res = results[0]

        # Extract detection data
        boxes = np.empty((0,4))
        scores = np.array([])
        classes = np.array([])
        texts = []

        if hasattr(res, "boxes") and len(res.boxes) > 0:
            try:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                detection_count += len(boxes)
            except Exception:
                boxes = np.array(res.boxes.xyxy)
                scores = np.array(res.boxes.conf)
                classes = np.array(res.boxes.cls)
                detection_count += len(boxes)

            # Extract text from license plates if OCR is enabled
            if ocr_reader and boxes.shape[0] > 0:
                for (x1, y1, x2, y2) in boxes.astype(int):
                    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        text = extract_text_from_crop(ocr_reader, crop)
                        texts.append(text)
                    else:
                        texts.append("")

        # Draw detections
        draw_boxes(frame, boxes, scores, classes, names, texts if ocr_reader else None)

        # Save crops if enabled
        if save_crops_toggle and boxes.shape[0] > 0:
            for i, ((x1,y1,x2,y2), conf, cls) in enumerate(zip(boxes.astype(int), scores, classes)):
                x1, y1, x2, y2 = max(0,x1), max(0,y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                text_suffix = f"_{texts[i].replace(' ', '_')}" if texts and i < len(texts) and texts[i] else ""
                fname = os.path.join(args.save_dir, f"frame{frame_id:06d}_det{i}_cls{int(cls)}_{conf:.2f}{text_suffix}.jpg")
                cv2.imwrite(fname, crop)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0

        # Add status information to frame
        info_text = [
            f"FPS: {fps:.1f}",
            f"Detections: {len(boxes)}",
            f"Total Found: {detection_count}",
            f"Confidence: {args.conf}",
        ]

        if ocr_reader:
            info_text.append("OCR: ON")
        if save_crops_toggle:
            info_text.append("Saving: ON")

        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Display frame
        cv2.imshow("License Plate Detection", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save current frame
            save_path = os.path.join(args.save_dir, f"frame_{frame_id:06d}_full.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Frame saved: {save_path}")
        elif key == ord("c"):
            # Toggle crop saving
            save_crops_toggle = not save_crops_toggle
            print(f"Crop saving: {'ON' if save_crops_toggle else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDetection completed. Total license plates found: {detection_count}")

if __name__ == "__main__":
    args = parse_args()
    try:
        run_camera(args)
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
