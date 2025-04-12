# Enhanced Manga Generator using TrinArt with image processing

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import our image utilities
from utils.image_utils import (
    apply_manga_style, enhance_manga_quality, 
    add_manga_tones, add_manga_effects,
    create_manga_panel_border, add_speed_lines
)

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced manga generation with TrinArt")
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
    
    # Post-processing options
    parser.add_argument("--style", type=str, default="standard", 
                        choices=["standard", "highcontrast", "sketch", "screentone", "action"],
                        help="Manga style to apply")
    parser.add_argument("--enhance", action="store_true", help="Apply quality enhancement")
    parser.add_argument("--contrast", type=float, default=1.3, help="Contrast enhancement factor")
    parser.add_argument("--sharpness", type=float, default=1.5, help="Sharpness enhancement factor")
    parser.add_argument("--border", action="store_true", help="Add manga panel border")
    parser.add_argument("--effect", type=str, default=None, 
                        choices=[None, "action", "emotional", "impact", "background"],
                        help="Special effect to apply")
    parser.add_argument("--speed_lines", action="store_true", help="Add speed lines")
    parser.add_argument("--speed_direction", type=str, default="radial",
                        choices=["horizontal", "vertical", "radial"],
                        help="Direction of speed lines")
    
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
        safety_checker=None  # Disable safety checker for manga generation
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
    
    print("Base image generated. Applying enhancements...")
    
    # Apply manga style
    if args.style != "standard":
        print(f"Applying {args.style} manga style...")
        image = apply_manga_style(image, style=args.style)
    
    # Apply quality enhancement
    if args.enhance:
        print("Enhancing image quality...")
        image = enhance_manga_quality(
            image, 
            contrast=args.contrast,
            sharpness=args.sharpness,
            brightness=1.1
        )
    
    # Add special effects
    if args.effect:
        print(f"Adding {args.effect} effect...")
        image = add_manga_effects(image, effect_type=args.effect)
    
    # Add speed lines
    if args.speed_lines:
        print(f"Adding {args.speed_direction} speed lines...")
        image = add_speed_lines(image, intensity=1.0, direction=args.speed_direction)
    
    # Add manga panel border
    if args.border:
        print("Adding panel border...")
        image = create_manga_panel_border(image, border_width=3)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Save the image
    image.save(args.output)
    print(f"Enhanced manga image saved to: {args.output}")

if __name__ == "__main__":
    main() 