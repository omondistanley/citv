import os
import cv2
import numpy as np
import argparse
import torch
from pathlib import Path
from depth import DepthEstimator, Segmentor
from config import PreprocessConfig

def main():
    parser = argparse.ArgumentParser(description="Run depth estimation and segmentation on images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Setup output subdirectories
    depth_out_dir = output_dir / "depth"
    seg_out_dir = output_dir / "segmentation"
    
    # Configuration
    config = PreprocessConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=(512, 512),
        save_depth_visualizations=True,
        save_depth_16bit=False
    )
    
    print(f"Running with device: {config.device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. Depth Estimation
    print("\n--- Testing Depth Estimation ---")
    estimator = None
    try:
        estimator = DepthEstimator(config)
        depths = estimator.estimate_depth(
            input_path=str(input_dir),
            output_dir=str(depth_out_dir),
            model_prefix="depth_anything_v2"
        )
        print(f"Successfully estimated depth for {len(depths)} frames.")
        
    except Exception as e:
        print(f"Depth estimation failed: {e}")

    # 2. Segmentation
    print("\n--- Testing Segmentation ---")
    try:
        segmentor = Segmentor(config)
        # Mock flow dir for fallback if needed, though we prefer model-based segmentation
        flow_dir = output_dir / "flow" # Assuming flow might be generated elsewhere or not needed if model works
        os.makedirs(flow_dir, exist_ok=True)
        
        segmentor.segment_frames(
            input_path=str(input_dir),
            flow_dir=str(flow_dir),
            output_dir=str(seg_out_dir)
        )
        print("Segmentation completed.")
    except Exception as e:
        print(f"Segmentation failed: {e}")

    # 3. Scene Understanding (New)
    print("\n--- Testing Scene Understanding (GRiT Integration) ---")
    if estimator is None:
        print("Skipping Scene Understanding because DepthEstimator failed to initialize.")
    else:
        try:
            from scene_understanding import SceneUnderstandingPipeline
            
            # Initialize the pipeline with the depth estimator we created earlier
            # This will also initialize GRiT, SGSG, and Pix2SG wrappers
            scene_pipe = SceneUnderstandingPipeline(estimator)
            
            # Process each image in the input directory
            scene_out_dir = Path(args.output_dir) / "scene_graph"
            scene_out_dir.mkdir(parents=True, exist_ok=True)
            
            input_p = Path(args.input_dir)
            images = []
            if input_p.is_file():
                images = [input_p]
            else:
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    images.extend(input_p.glob(ext))
            
            if not images:
                print(f"No images found in {input_p}")
            
            for img_path in images:
                print(f"Processing {img_path.name} with GRiT...")
                scene_pipe.process_image(str(img_path), str(scene_out_dir))
                
            print(f"Scene understanding completed. Results saved to {scene_out_dir}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Scene understanding failed: {e}")

if __name__ == "__main__":
    main()
