import os
import argparse
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from manga_prompt_generator import MangaPromptGenerator
from utils.image_utils import (
    apply_manga_style, 
    enhance_manga_quality, 
    add_manga_effects, 
    create_manga_panel_border, 
    add_speed_lines,
    add_manga_tones,
    create_manga_layout
)
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-panel manga story")
    # Story parameters
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Main story prompt/description for the manga")
    parser.add_argument("--panels", type=int, default=4,
                        help="Number of panels to generate")
    parser.add_argument("--style", type=str, default="manga",
                        choices=["manga", "shonen", "seinen", "shoujo", "action", "noir"],
                        help="Style of manga to generate")
    parser.add_argument("--layout", type=str, default="vertical",
                        choices=["vertical", "horizontal", "grid"],
                        help="Layout type for the manga panels")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=30, 
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for generation")
    parser.add_argument("--width", type=int, default=512, 
                        help="Image width in pixels")
    parser.add_argument("--height", type=int, default=512, 
                        help="Image height in pixels")
    
    # Context handling
    parser.add_argument("--context_strength", type=float, default=0.3,
                        help="Strength of influence from previous panels (0-1)")
    parser.add_argument("--use_text_context", action="store_true",
                        help="Include previous panel descriptions in prompts")
    
    # Model parameters
    parser.add_argument("--llm_model", type=str, default="llama3-70b-8192",
                        help="Groq LLM model to use for prompt generation")
    parser.add_argument("--diffusion_model", type=str, default="diffusers-115k",
                        choices=["diffusers-60k", "diffusers-95k", "diffusers-115k"],
                        help="TrinArt model version")
    
    # Post-processing parameters
    parser.add_argument("--manga_style", type=str, default="highcontrast",
                        choices=["highcontrast", "sketch", "screentone", "action"],
                        help="Manga style to apply to generated images")
    parser.add_argument("--enhance", action="store_true", 
                        help="Enhance image quality")
    parser.add_argument("--effects", type=str, default=None,
                        choices=[None, "action", "emotional", "impact", "background"],
                        help="Special effects to apply")
    parser.add_argument("--tones", action="store_true", 
                        help="Add manga tones to the images")
    parser.add_argument("--borders", action="store_true", 
                        help="Add borders to panels")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for generated images")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual panels as well as the combined layout")
    
    return parser.parse_args()

def setup_pipeline(device, model_version):
    """Set up and return the StableDiffusionPipeline"""
    model_id = "naclbit/trinart_stable_diffusion_v2"
    
    print(f"Loading TrinArt model ({model_version})...")
    
    try:
        # Try loading the pipeline with a specific revision
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision=model_version,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for manga
            requires_safety_checker=False
        )
    except Exception as e:
        print(f"Error loading specific revision: {e}")
        print("Trying to load the model without specific revision...")
        # Fallback to loading without specific revision
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Enable memory optimization if on CUDA
    if device == "cuda":
        pipeline.enable_attention_slicing()
        
    return pipeline

def generate_panel_image(pipeline, prompt, args, panel_index, previous_panels=None, previous_descriptions=None, generator=None):
    """
    Generate a single panel image using the provided prompt and contextual information from previous panels
    
    Args:
        pipeline: StableDiffusionPipeline
        prompt: Panel-specific prompt
        args: Command line arguments
        panel_index: Index of the current panel (0-based)
        previous_panels: List of previously generated PIL Images
        previous_descriptions: List of previous panel text descriptions
        generator: Torch random generator
    
    Returns:
        PIL.Image: Generated panel image
    """
    # Build context-aware prompt by incorporating previous panel descriptions
    enhanced_prompt = prompt
    
    if args.use_text_context and previous_descriptions and panel_index > 0:
        # Make sure we have a string for the previous description
        if previous_descriptions and len(previous_descriptions) > 0:
            prev_desc = previous_descriptions[-1]
            if not isinstance(prev_desc, str):
                prev_desc = str(prev_desc)
                
            # Truncate previous description to avoid overly long prompts
            max_context_length = 50  # Limiting context to around 50 chars to leave room for the main prompt
            context_text = f"Previous panel: {prev_desc[:max_context_length]}... "
            enhanced_prompt = context_text + enhanced_prompt
        
    # Further enhance prompt with manga-specific terms
    style_suffix = f", manga style, {args.style} manga panel, detailed manga art"
    
    # Estimate token length and truncate if needed
    # CLIP can handle ~77 tokens, with each token being roughly 4 chars on average
    # Reserve about 20 tokens (80 chars) for the style suffix
    max_prompt_length = 200  # Approximately 50 tokens, leaving room for the style suffix
    if len(enhanced_prompt) > max_prompt_length:
        enhanced_prompt = enhanced_prompt[:max_prompt_length] + "..."
        
    enhanced_prompt = enhanced_prompt + style_suffix
    
    # Negative prompt to avoid common issues
    negative_prompt = "low quality, blurry, distorted, bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, too many fingers, long neck, mutation, deformed, ugly, poorly drawn face, cloned face, unrealistic, inconsistent style"
    
    print(f"Generating panel {panel_index + 1}/{args.panels}...")
    print(f"Prompt length: {len(enhanced_prompt)} chars")
    
    # Generate the image
    with torch.no_grad():
        result = pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            width=args.width,
            height=args.height
        )
        
    image = result.images[0]
    
    # Apply image-based context if we have previous panels and context strength > 0
    if previous_panels and panel_index > 0 and args.context_strength > 0:
        # Use the most recent previous panel for visual continuity
        prev_image = previous_panels[-1]
        image = apply_visual_context(image, prev_image, args.context_strength)
    
    # Apply post-processing based on args
    if args.manga_style:
        image = apply_manga_style(image, style=args.manga_style)
    
    if args.enhance:
        image = enhance_manga_quality(image)
    
    if args.effects:
        image = add_manga_effects(image, effect_type=args.effects)
    
    if args.tones:
        image = add_manga_tones(image, pattern_type="dots" if args.manga_style != "screentone" else "crosshatch")
    
    if args.borders:
        image = create_manga_panel_border(image)
    
    return image

def apply_visual_context(current_image, previous_image, strength=0.3):
    """
    Apply visual context from the previous panel to maintain consistency
    
    This works by slightly blending color palettes and tones between panels
    for visual continuity
    
    Args:
        current_image: Current panel image
        previous_image: Previous panel image
        strength: Strength of the context influence (0-1)
    
    Returns:
        PIL.Image: Image with context applied
    """
    # Resize previous image to match current if needed
    if previous_image.size != current_image.size:
        previous_image = previous_image.resize(current_image.size, Image.LANCZOS)
    
    # Convert to numpy arrays
    curr_array = np.array(current_image).astype(float)
    prev_array = np.array(previous_image).astype(float)
    
    # Apply subtle blending for consistency
    # This creates visual continuity without copying the previous image
    result_array = (1 - strength) * curr_array + strength * prev_array
    
    # Clip to valid range and convert back to uint8
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_array)

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    try:
        # Step 1: Generate panel prompts using Groq
        print("Generating panel prompts with Groq LLM...")
        prompt_generator = MangaPromptGenerator()
        
        try:
            panel_prompts = prompt_generator.generate_panel_prompts(
                story_prompt=args.prompt,
                num_panels=args.panels,
                style=args.style,
                model=args.llm_model
            )
            
            print(f"Raw panel_prompts type: {type(panel_prompts)}")
            
            # Handle case where panel_prompts might not be a list
            if not isinstance(panel_prompts, list):
                print(f"Warning: panel_prompts is not a list but {type(panel_prompts)}")
                if isinstance(panel_prompts, str):
                    # If it's a single string, try to split it into multiple panels
                    panel_prompts = panel_prompts.split("\n\n")
                    print(f"Split string into {len(panel_prompts)} panels")
                elif isinstance(panel_prompts, dict):
                    # If it's a dict, try to extract values
                    panel_prompts = list(panel_prompts.values())
                    print(f"Converted dict to list with {len(panel_prompts)} panels")
                else:
                    # Last resort: convert to string and split
                    panel_prompts = str(panel_prompts).split("\n\n")
                    print(f"Converted to string and split into {len(panel_prompts)} panels")
            
            # Ensure we have the right number of panels
            if len(panel_prompts) < args.panels:
                print(f"Warning: Only got {len(panel_prompts)} panels, adding generic ones")
                for i in range(len(panel_prompts), args.panels):
                    panel_prompts.append(f"Manga panel {i+1} in {args.style} style")
            elif len(panel_prompts) > args.panels:
                print(f"Warning: Got {len(panel_prompts)} panels, truncating to {args.panels}")
                panel_prompts = panel_prompts[:args.panels]
                
            # Ensure all prompts are strings with proper length
            for i in range(len(panel_prompts)):
                if not isinstance(panel_prompts[i], str):
                    print(f"Warning: Converting non-string prompt at index {i} from {type(panel_prompts[i])} to string")
                    panel_prompts[i] = str(panel_prompts[i])
                
                # Limit prompt length
                if len(panel_prompts[i]) > 300:
                    print(f"Warning: Truncating overly long prompt at index {i}")
                    panel_prompts[i] = panel_prompts[i][:300] + "..."
                
                # Ensure minimum content
                if not panel_prompts[i].strip():
                    panel_prompts[i] = f"Manga panel {i+1} in {args.style} style"
                    print(f"Warning: Empty prompt at index {i}, using generic prompt")
                
                print(f"Panel {i+1} length: {len(panel_prompts[i])} chars")
            
        except Exception as prompt_error:
            print(f"Error generating panel prompts: {prompt_error}")
            print("Falling back to generic panel prompts")
            # Create generic panel prompts as fallback
            panel_prompts = []
            for i in range(args.panels):
                panel_prompts.append(f"Panel {i+1} of {args.prompt} in {args.style} manga style")
        
        # Save prompts to file
        prompts_file = os.path.join(args.output_dir, "panel_prompts.json")
        try:
            prompt_generator.save_prompts_to_file(panel_prompts, prompts_file)
            print(f"Saved panel prompts to {prompts_file}")
        except Exception as save_error:
            print(f"Warning: Could not save prompts to file: {save_error}")
        
        # Print the prompts for verification
        print("\nPanel prompts generated:")
        for i, prompt in enumerate(panel_prompts, 1):
            print(f"Panel {i}: {prompt[:100]}...")
        
        # Step 2: Generate images for each panel
        pipeline = setup_pipeline(device, args.diffusion_model)
        
        panel_images = []
        
        for i, prompt in enumerate(panel_prompts):
            # Create panel-specific generator for reproducibility
            panel_generator = None
            if generator:
                panel_generator = torch.Generator(device=device).manual_seed(args.seed + i)
                
            # Generate each panel with context from previous panels
            try:
                image = generate_panel_image(
                    pipeline=pipeline, 
                    prompt=prompt, 
                    args=args, 
                    panel_index=i,
                    previous_panels=panel_images if i > 0 else None,
                    previous_descriptions=panel_prompts[:i] if i > 0 else None,
                    generator=panel_generator
                )
                
                panel_images.append(image)
                
                # Save individual panel if requested
                if args.save_individual:
                    panel_path = os.path.join(args.output_dir, f"panel_{i+1}.png")
                    image.save(panel_path)
                    print(f"Saved panel {i+1} to {panel_path}")
            except Exception as panel_error:
                print(f"Error generating panel {i+1}: {panel_error}")
                # If we failed to generate this panel, create a blank panel as fallback
                if i == 0 or not panel_images:
                    # Create a blank image for the first panel
                    blank_image = Image.new('RGB', (args.width, args.height), color='white')
                    panel_images.append(blank_image)
                else:
                    # Copy the previous panel as fallback
                    panel_images.append(panel_images[-1].copy())
                
                if args.save_individual:
                    panel_path = os.path.join(args.output_dir, f"panel_{i+1}_error.png")
                    panel_images[-1].save(panel_path)
                    print(f"Saved error fallback for panel {i+1} to {panel_path}")
        
        # Step 3: Create manga layout with all panels
        print("Creating manga layout...")
        manga_layout = create_manga_layout(
            panel_images, 
            layout_type=args.layout,
            margin=10
        )
        
        # Save final manga layout
        layout_path = os.path.join(args.output_dir, f"manga_story_{args.style}.png")
        manga_layout.save(layout_path)
        print(f"Saved complete manga layout to {layout_path}")
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 