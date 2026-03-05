# Project Setup and Run Guide

## SSH into GCP Instance

To SSH into the Google Cloud instance, run the provided helper script:

```bash
./ssh_gcp.sh
```

Or manually run:

```bash
gcloud compute ssh --zone "zone here" "instance here" --project "name"
```

## Running the Project

Once connected to the instance, you can run the scene understanding pipeline using `test_depth.py`.

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Usage

Run the pipeline on a directory of images:

```bash
python test_depth.py --input_dir images --output_dir output
```

Arguments:
- `--input_dir`: Directory containing input images (e.g., `images/`).
- `--output_dir`: Directory to save results (default: `output/`).

The pipeline performs:
1. Depth Estimation (Depth Anything V2)
2. Segmentation
3. Scene Understanding (GRiT + SGSG + Pix2SG integration)
