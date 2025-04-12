#!/usr/bin/env python3
"""
Manga Panel Arrangement Utility
This script arranges individual image panels into manga page layouts.
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from PIL import Image

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.image_utils import create_comic_page, create_manga_layout, add_text_bubble

def parse_args():
    parser = argparse.ArgumentParser(description="Arrange manga panels into layouts")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input directory containing panel images or a specific pattern (e.g. 'panels/*.png')")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image file path")
    parser.add_argument("--layout", type=str, default="vertical",
                        choices=["vertical", "horizontal", "grid", "custom"],
                        help="Layout type for arranging panels")
    parser.add_argument("--layout-file", type=str,
                        help="JSON file with custom layout specifications (required if --layout=custom)")
    parser.add_argument("--width", type=int, default=1200,
                        help="Page width in pixels")
    parser.add_argument("--height", type=int, default=1600,
                        help="Page height in pixels")
    parser.add_argument("--margin", type=int, default=10,
                        help="Margin between panels in pixels")
    parser.add_argument("--sort", type=str, default="numeric",
                        choices=["numeric", "alphabetic", "time", "none"],
                        help="Method for sorting input files")
    parser.add_argument("--border", action="store_true",
                        help="Add borders to panels")
    parser.add_argument("--manga-style", action="store_true",
                        help="Apply manga-style reading order (right-to-left)")
    parser.add_argument("--dialogue-file", type=str,
                        help="JSON file with dialogue text for panels")
    return parser.parse_args()

def find_panel_images(input_pattern, sort_method="numeric"):
    """Find and sort panel images based on the input pattern and sorting method"""
    # Handle directory vs pattern
    if os.path.isdir(input_pattern):
        # If input is a directory, find all image files
        image_files = []
        for ext in ["png", "jpg", "jpeg", "webp"]:
            image_files.extend(glob.glob(os.path.join(input_pattern, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(input_pattern, f"*.{ext.upper()}")))
    else:
        # If input is a pattern, use it directly
        image_files = glob.glob(input_pattern)
    
    if not image_files:
        raise ValueError(f"No image files found for pattern: {input_pattern}")
    
    # Sort files based on the specified method
    if sort_method == "numeric":
        # Sort numerically (extract numbers from filenames)
        def extract_number(filename):
            parts = os.path.basename(filename).split('_')
            for part in parts:
                digits = ''.join(filter(str.isdigit, part))
                if digits:
                    return int(digits)
            return 0
        
        image_files.sort(key=extract_number)
    elif sort_method == "alphabetic":
        # Sort alphabetically by filename
        image_files.sort()
    elif sort_method == "time":
        # Sort by file modification time
        image_files.sort(key=os.path.getmtime)
    # "none" means keep the order from glob
    
    return image_files

def load_custom_layout(layout_file):
    """Load custom layout specifications from a JSON file"""
    if not os.path.exists(layout_file):
        raise FileNotFoundError(f"Layout file not found: {layout_file}")
    
    with open(layout_file, 'r') as f:
        layout_data = json.load(f)
    
    # Validate layout data
    if not isinstance(layout_data, list):
        raise ValueError("Layout file must contain a list of panel specifications")
    
    for panel in layout_data:
        if not all(key in panel for key in ["x", "y", "w", "h"]):
            raise ValueError("Each panel must have x, y, w, h coordinates (normalized 0-1)")
    
    # Extract normalized coordinates
    return [(panel["x"], panel["y"], panel["w"], panel["h"]) for panel in layout_data]

def load_dialogue(dialogue_file):
    """Load dialogue data from a JSON file"""
    if not dialogue_file or not os.path.exists(dialogue_file):
        return None
    
    with open(dialogue_file, 'r') as f:
        dialogue_data = json.load(f)
    
    return dialogue_data

def apply_dialogue_to_panels(panel_images, dialogue_data):
    """Apply dialogue to panel images"""
    if not dialogue_data:
        return panel_images
    
    result_panels = []
    
    for i, image in enumerate(panel_images):
        panel_idx = str(i)
        if panel_idx in dialogue_data:
            dialogue = dialogue_data[panel_idx]
            
            # Check if dialogue has position information
            if isinstance(dialogue, dict) and "text" in dialogue:
                text = dialogue["text"]
                position = dialogue.get("position", (image.width // 2, image.height // 2))
                bubble_type = dialogue.get("type", "speech")
                image_with_text = add_text_bubble(image, text, position, bubble_type=bubble_type)
            else:
                # Simple dialogue text
                image_with_text = add_text_bubble(image, dialogue, (image.width // 2, image.height // 2))
            
            result_panels.append(image_with_text)
        else:
            result_panels.append(image)
    
    return result_panels

def main():
    args = parse_args()
    
    try:
        # Find panel images
        image_files = find_panel_images(args.input, args.sort)
        print(f"Found {len(image_files)} panel images")
        
        # Load images
        panel_images = [Image.open(file) for file in image_files]
        
        # Apply manga-style reading order if requested
        if args.manga_style and args.layout != "custom":
            panel_images.reverse()
        
        # Load dialogue if specified
        dialogue_data = load_dialogue(args.dialogue_file)
        if dialogue_data:
            print(f"Applying dialogue from {args.dialogue_file}")
            panel_images = apply_dialogue_to_panels(panel_images, dialogue_data)
        
        # Create layout based on the specified method
        if args.layout == "custom":
            if not args.layout_file:
                raise ValueError("--layout-file is required when using custom layout")
            
            custom_layout = load_custom_layout(args.layout_file)
            print(f"Using custom layout with {len(custom_layout)} panel positions")
            
            # Ensure we have enough layout positions
            if len(custom_layout) < len(panel_images):
                print(f"Warning: More panels ({len(panel_images)}) than layout positions ({len(custom_layout)})")
                panel_images = panel_images[:len(custom_layout)]
            elif len(custom_layout) > len(panel_images):
                print(f"Warning: More layout positions ({len(custom_layout)}) than panels ({len(panel_images)})")
                custom_layout = custom_layout[:len(panel_images)]
            
            page = create_comic_page(
                panel_images,
                layout=custom_layout,
                page_width=args.width,
                page_height=args.height,
                margin=args.margin
            )
        else:
            # For standard layouts
            page = create_manga_layout(
                panel_images,
                layout_type=args.layout,
                margin=args.margin,
                bg_color=255
            )
            
            # Resize to target dimensions if needed
            current_width, current_height = page.size
            scale_factor = min(args.width / current_width, args.height / current_height)
            new_width = int(current_width * scale_factor)
            new_height = int(current_height * scale_factor)
            
            if scale_factor != 1.0:
                page = page.resize((new_width, new_height), Image.LANCZOS)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Save the result
        page.save(args.output)
        print(f"Saved manga layout to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()