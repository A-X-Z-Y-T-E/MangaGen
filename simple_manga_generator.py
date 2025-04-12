import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple manga generation with TrinArt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image")
    parser.add_argument("--output", type=str, default="manga_output.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels")
    parser.add_argument("--model_version", type=str, default="diffusers-115k", 
                        choices=["diffusers-60k", "diffusers-95k", "diffusers-115k"],
                        help="TrinArt model version (60k for lighter stylization, 115k for stronger)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set model parameters
    model_id = "naclbit/trinart_stable_diffusion_v2"
    revision = args.model_version
    
    print(f"Loading TrinArt model ({revision})...")
    
    # Load the model
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        #safety_checker=None  # Disable safety checker for manga generation
    )
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Enable memory optimization if on CUDA
    if device == "cuda":
        pipeline.enable_attention_slicing()
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Enhance prompt with manga-specific terms
    enhanced_prompt = f"{args.prompt}, manga style, black and white manga panel, detailed manga art"
    
    # Negative prompt to avoid common issues
    negative_prompt = "low quality, blurry, distorted, bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, too many fingers, long neck, mutation, deformed, ugly, poorly drawn face, cloned face, unrealistic"
    
    print(f"Generating image with prompt: {args.prompt}")
    
    # Generate the image
    with torch.no_grad():
        image = pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            width=args.width,
            height=args.height
        ).images[0]
    
    # Save the image
    image.save(args.output)
    print(f"Image saved to: {args.output}")

if __name__ == "__main__":
    main() 