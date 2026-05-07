from ultralytics import YOLO

def export_to_tensorrt(model_path="weapon_model.pt", img_size=640):
    print(f"Exporting {model_path} to TensorRT...")
    try:
        model = YOLO(model_path)
        # Export the model to TensorRT format
        # This will create 'weapon_model.engine' in the current directory
        # Requires tensorrt to be installed
        model.export(format="engine", imgsz=img_size, dynamic=False, simplify=True, half=True)
        print("Export completed successfully.")
        print("TensorRT Engine saved for Jetson Nano deployment.")
    except Exception as e:
        print(f"Error during export: {e}")
        print("Make sure TensorRT and ONNX are installed: pip install onnx onnxruntime tensorrt")

if __name__ == "__main__":
    export_to_tensorrt()
