"""
Train YOLOv8 model for license plate detection using the Roboflow dataset.
This script will create the best.pt file needed for detection.

Usage:
    python train_model.py --epochs 30 --imgsz 640 --batch 16
"""
import argparse
import os
import yaml
from pathlib import Path
# New: optional torch import to decide GPU/CPU automatically
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# New: device resolver and torch summary
def _resolve_device(dev_arg: str) -> str:
    # If user provided a device, normalize and validate
    if dev_arg:
        if dev_arg == '0':
            desired = 'cuda:0'
        elif dev_arg.lower() in ('cuda', 'gpu'):
            desired = 'cuda:0'
        else:
            desired = dev_arg
        if desired.startswith('cuda'):
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                return desired
            print("Warning: CUDA requested but not available in PyTorch. Falling back to CPU.")
            return 'cpu'
        return desired
    # Auto: prefer CUDA when usable by PyTorch
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'

def _print_torch_info():
    if not _TORCH_AVAILABLE:
        print("PyTorch not installed; Ultralytics will attempt its own device handling (likely CPU).")
        return
    try:
        print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | CUDA build: {getattr(torch.version, 'cuda', None)}")
        if torch.cuda.is_available():
            print(f"GPU(s): {torch.cuda.device_count()} | Device 0: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 License Plate Detection Model')
    parser.add_argument('--data', type=str, default='data.yaml', help='path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')  # was 100
    parser.add_argument('--imgsz', type=int, default=640, help='image size for training')
    parser.add_argument('--batch', type=int, default=16, help='batch size (adjust based on GPU memory)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='pretrained model to start from')
    parser.add_argument('--name', type=str, default='license_plate_detection', help='experiment name')
    parser.add_argument('--device', type=str, default='', help='device to train on (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')  # was 50
    return parser.parse_args()

def check_dataset_structure():
    """Check if the dataset structure matches data.yaml configuration"""
    print("Checking dataset structure...")

    # Load data.yaml
    if not os.path.exists('data.yaml'):
        raise FileNotFoundError("data.yaml not found. Make sure you're in the correct directory.")

    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # Check paths
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')

    print(f"Train path: {train_path}")
    print(f"Validation path: {val_path}")

    # Check if paths exist
    issues = []

    if not os.path.exists(train_path):
        issues.append(f"Train images directory not found: {train_path}")
    else:
        train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {train_images} training images")
        if train_images == 0:
            issues.append("No training images found")

    if not os.path.exists(val_path):
        issues.append(f"Validation images directory not found: {val_path}")
    else:
        val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {val_images} validation images")
        if val_images == 0:
            issues.append("No validation images found")

    # Check for label directories
    train_labels = train_path.replace('images', 'labels')
    val_labels = val_path.replace('images', 'labels')

    if not os.path.exists(train_labels):
        issues.append(f"Train labels directory not found: {train_labels}")
    else:
        train_label_count = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
        print(f"Found {train_label_count} training labels")

    if not os.path.exists(val_labels):
        issues.append(f"Validation labels directory not found: {val_labels}")
    else:
        val_label_count = len([f for f in os.listdir(val_labels) if f.endswith('.txt')])
        print(f"Found {val_label_count} validation labels")

    if issues:
        print("\nDataset Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✓ Dataset structure looks good!")
    return True

def train_model(args):
    """Train the YOLOv8 model"""
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLOv8 imported successfully")
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    # New: show torch/CUDA status and resolve device (defaults to CUDA if available)
    _print_torch_info()
    device_str = _resolve_device(args.device)
    print(f"Selected training device: {device_str}")

    # Check dataset first
    if not check_dataset_structure():
        print("\nPlease fix the dataset issues before training.")
        return

    print(f"\nInitializing YOLOv8 model: {args.model}")
    model = YOLO(args.model)  # This will download the pretrained model if needed

    # New: ensure runs are saved inside this repo (not CWD of IDE)
    project_dir = (Path(__file__).parent / "runs").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training outputs will be saved under: {project_dir}")

    print(f"Starting training with the following parameters:")
    print(f"  - Data: {args.data}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Device: {device_str}")
    print(f"  - Workers: {args.workers}")

    # Start training (device now explicitly set)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=device_str,  # changed: always pass resolved device
        workers=args.workers,
        patience=args.patience,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True,
        project=str(project_dir),    # New: force outputs into <repo>/runs
        exist_ok=True                # New: allow reuse/increment under same name
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)

    # New: resolve actual save_dir from trainer if available
    save_dir = None
    try:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            save_dir = Path(save_dir)
    except Exception:
        save_dir = None

    # Fallback: look in both runs/detect and runs/train under our project runs
    latest_run = None
    if save_dir and save_dir.exists():
        latest_run = save_dir
    else:
        candidates = []
        for sub in ("detect", "train"):
            subdir = project_dir / sub
            if subdir.exists():
                for p in subdir.glob(f"{args.name}*"):
                    candidates.append(p)
        if candidates:
            latest_run = max(candidates, key=os.path.getctime)

    if latest_run:
        best_pt = latest_run / 'weights' / 'best.pt'
        last_pt = latest_run / 'weights' / 'last.pt'

        print(f"Training results saved to: {latest_run}")
        print(f"Best model weights: {best_pt}")
        print(f"Last model weights: {last_pt}")

        import shutil
        if best_pt.exists():
            shutil.copy2(best_pt, Path(__file__).parent / 'best.pt')
            print(f"✓ Copied best.pt to: {(Path(__file__).parent / 'best.pt').resolve()}")

            # Validate the model
            print("\nValidating model performance...")
            model = YOLO(str(Path(__file__).parent / 'best.pt'))
            val_results = model.val()

            print(f"\nModel Performance:")
            print(f"  - mAP50: {val_results.box.map50:.4f}")
            print(f"  - mAP50-95: {val_results.box.map:.4f}")

        print(f"\n✓ Training complete! You can now run detection with:")
        print(f"  python detect_camera.py --weights best.pt")
    else:
        print(f"Warning: Could not locate training output under: {project_dir}")
        print("Tip: Check your console for Ultralytics 'save_dir' path or set a unique --name.")

def main():
    args = parse_args()

    print("YOLOv8 License Plate Detection - Model Training")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists('data.yaml'):
        print("Error: data.yaml not found in current directory")
        print("Please run this script from the dataset directory")
        return

    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Reduce batch size if you get out of memory errors")
        print("2. Check that your dataset paths in data.yaml are correct")
        print("3. Ensure you have enough disk space for training outputs")
        print("4. Try with CPU training if GPU issues: --device cpu")

if __name__ == "__main__":
    main()
