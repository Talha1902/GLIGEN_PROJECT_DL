import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the stable diffusion inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16
).to(device)

# Function to load an image from a file and return a PIL Image
def load_image_from_file(file_path):
    img = Image.open(file_path).convert("RGB")
    return img

# Function to generate and save synthetic defected images
def generate_synthetic_defect(image_path, mask_image_path, prompt, save_path):
    # Load the defect-free image and mask image
    image = load_image_from_file(image_path)
    mask_image = load_image_from_file(mask_image_path)

    # Generate the modified image with synthetic defects
    with torch.no_grad():
        images = pipe(prompt=prompt, image=image, mask_image=mask_image).images

    # Save the modified image
    images[0].save(save_path)
    print(f"Generated defected image saved to {save_path}")

# Define paths to the dataset directories
base_dir = 'path_to_kolektor_dataset'  # Change this to the base directory of your dataset
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
output_dir = 'path_to_save_defected_images'  # Directory to save generated defected images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over train and test directories
for split in ['train', 'test']: 
    image_dir = os.path.join(base_dir, split)
    
    for file_name in os.listdir(image_dir):
        # Process only original images with the _GT suffix
        if '_GT.jpg' in file_name:
            # Remove the _GT suffix to get the base name
            base_name = file_name.replace('_GT.jpg', '')

            # Get paths for the images
            normal_image_path = os.path.join(image_dir, file_name)
            mask_image_path = os.path.join(image_dir, f"{base_name}_mask.jpg")  # Assumed mask naming convention
            defected_image_path = os.path.join(output_dir, f"{base_name}.jpg")

            # Check if all paths exist
            if os.path.exists(normal_image_path) and os.path.exists(mask_image_path):
                # Define prompts for generating synthetic defects
                prompt = "add a scratch on the surface"  # You can modify this prompt as needed
                generate_synthetic_defect(normal_image_path, mask_image_path, prompt, defected_image_path)
