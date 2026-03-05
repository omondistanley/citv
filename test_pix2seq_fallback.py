import cv2
import torch
import numpy as np
from scene_understanding import Pix2SeqWrapper

def test_pix2seq_fallback():
    print("Testing Pix2Seq / OWL-ViT fallback...")
    
    # Create dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize wrapper
    device = torch.device("cpu")
    wrapper = Pix2SeqWrapper(device)
    
    if wrapper.model_type == "owlvit":
        print("Success: Wrapper correctly fell back to OWL-ViT.")
    elif wrapper.model_type == "pix2seq":
        print("Note: Wrapper found official Pix2Seq checkpoint.")
    else:
        print("Error: Wrapper failed to initialize either model.")
        
    # Run dummy prediction
    detections = wrapper.predict(img)
    print(f"Prediction run successful. Detections found: {len(detections)}")

if __name__ == "__main__":
    test_pix2seq_fallback()
