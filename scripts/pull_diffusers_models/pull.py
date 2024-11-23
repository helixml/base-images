import argparse
import os

from huggingface_hub import snapshot_download


def download_model(model_name: str):
    """
    Download model weights from Hugging Face Hub
    
    Args:
        model_name (str): Name of the model on Hugging Face Hub
        save_path (str): Local directory path to save the model
        pipeline_type (str): Type of pipeline to use ('sd' for StableDiffusion or 'flux' for Flux)
    """
    print(f"Downloading model: {model_name}")
    
    # Download all model files directly without pipeline initialization
    snapshot_download(
        repo_id=model_name,
        ignore_patterns=["*.md", "*.txt"],
    )

    # Check the location of the downloaded models
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    print(f"Model successfully downloaded to: {cache_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Stable Diffusion or Flux model weights")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model on Hugging Face Hub (e.g., 'runwayml/stable-diffusion-v1-5' or 'black-forest-labs/FLUX.1-schnell')"
    )
    
    args = parser.parse_args()
    download_model(args.model_name)
