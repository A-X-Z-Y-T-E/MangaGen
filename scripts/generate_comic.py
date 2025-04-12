#!/usr/bin/env python3
"""
Comic Generation Wrapper Script
This script provides a simplified interface to generate multi-page comics.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manga_prompt_generator import MangaPromptGenerator
from generate_manga_story import main as generate_manga
from utils.image_utils import create_comic_page, add_text_bubble, save_images

def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-page comic/manga")
    parser.add_argument("--title", type=str, required=True, 
                        help="Comic title or high-level concept")
    parser.add_argument("--pages", type=int, default=3,
                        help="Number of pages to generate")
    parser.add_argument("--panels-per-page", type=int, default=4,
                        help="Number of panels per page")
    parser.add_argument("--style", type=str, default="manga",
                        choices=["manga", "shonen", "seinen", "shoujo", "action", "noir", "american", "european"],
                        help="Comic style")
    parser.add_argument("--dialogue", action="store_true",
                        help="Add speech bubbles and dialogue")
    parser.add_argument("--output-dir", type=str, default="output/comic",
                        help="Output directory")
    parser.add_argument("--config", type=str,
                        help="Optional JSON config file with additional parameters")
    return parser.parse_args()

def generate_story_outline(title, pages, panels_per_page, style):
    """Generate an overall story outline and page descriptions"""
    prompt_generator = MangaPromptGenerator()
    
    # First generate high-level story arc
    story_arc_prompt = f"""
    Create a compelling story outline for a {style} comic/manga titled '{title}'.
    The story should be structured across {pages} pages with {panels_per_page} panels per page.
    Focus on narrative flow, character development, and visual storytelling.
    
    Provide a high-level summary and then a brief description for each page.
    """
    
    story_outline = prompt_generator.generate_panel_prompts(
        story_prompt=story_arc_prompt,
        num_panels=pages,  # One description per page
        style=style,
        model="llama3-70b-8192"
    )
    
    # Format as a structured story outline
    story = {
        "title": title,
        "style": style,
        "summary": story_outline[0] if len(story_outline) > 0 else title,
        "pages": []
    }
    
    # Add page outlines
    for i, page_desc in enumerate(story_outline[1:pages+1]):
        story["pages"].append({
            "page_number": i+1,
            "description": page_desc,
            "panels": []
        })
    
    return story

def generate_page_panels(story, page_idx, panels_per_page, style):
    """Generate panel descriptions for a specific page"""
    prompt_generator = MangaPromptGenerator()
    page = story["pages"][page_idx]
    
    # Create a prompt for panel generation based on page description
    panels_prompt = f"""
    For page {page_idx+1} of a {style} comic/manga about: {story['summary']}
    
    Page description: {page['description']}
    
    Create {panels_per_page} sequential panel descriptions that tell this part of the story.
    Each panel should be visually interesting and advance the narrative.
    Focus on composition, character positions, expressions, and key visual elements.
    """
    
    panel_descriptions = prompt_generator.generate_panel_prompts(
        story_prompt=panels_prompt,
        num_panels=panels_per_page,
        style=style,
        model="llama3-70b-8192"
    )
    
    # Update the story structure with panel info
    for i, panel_desc in enumerate(panel_descriptions):
        page["panels"].append({
            "panel_number": i+1,
            "description": panel_desc,
            "image_path": None,
            "dialogue": None
        })
    
    return page["panels"]

def generate_dialogue(story, page_idx, panel_idx, style):
    """Generate dialogue for a specific panel"""
    prompt_generator = MangaPromptGenerator()
    panel = story["pages"][page_idx]["panels"][panel_idx]
    
    dialogue_prompt = f"""
    For this {style} comic/manga panel:
    
    {panel['description']}
    
    Generate appropriate dialogue for this scene. Consider:
    1. Character emotions and personalities
    2. Plot advancement
    3. Visual context of the panel
    
    Provide just the dialogue text that would appear in a speech bubble.
    Keep it concise and impactful, as in real manga.
    """
    
    dialogue = prompt_generator.generate_panel_prompts(
        story_prompt=dialogue_prompt,
        num_panels=1,
        style=style,
        model="llama3-70b-8192"
    )
    
    # Use the first response as dialogue
    return dialogue[0] if dialogue else None

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config file if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Generate story outline
    print(f"Generating story outline for '{args.title}'...")
    story = generate_story_outline(args.title, args.pages, args.panels_per_page, args.style)
    
    # Save story outline
    with open(output_dir / "story_outline.json", 'w') as f:
        json.dump(story, f, indent=2)
    
    # Generate each page
    for page_idx in range(len(story["pages"])):
        page = story["pages"][page_idx]
        page_dir = output_dir / f"page_{page_idx+1}"
        page_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating page {page_idx+1} of {len(story['pages'])}")
        print(f"Description: {page['description'][:100]}...")
        
        # Generate panel descriptions for this page
        panels = generate_page_panels(story, page_idx, args.panels_per_page, args.style)
        
        # Generate each panel
        panel_images = []
        for panel_idx, panel in enumerate(panels):
            print(f"  Generating panel {panel_idx+1} - {panel['description'][:50]}...")
            
            # Prepare arguments for manga generation
            panel_args = [
                "--prompt", panel["description"],
                "--output", str(page_dir / f"panel_{panel_idx+1}.png"),
                "--style", args.style,
                "--manga_style", "highcontrast",
                "--enhance",
                "--borders",
                "--save_individual"
            ]
            
            # Add any extra parameters from config
            if "panel_params" in config:
                for param, value in config["panel_params"].items():
                    panel_args.extend([f"--{param}", str(value)])
            
            # Generate the panel
            try:
                # This would normally call the generate_manga function
                # For this script we'll just print a placeholder
                print(f"    Would generate panel with: {' '.join(panel_args)}")
                
                # In a real implementation:
                # generate_manga(panel_args)
                
                panel_path = page_dir / f"panel_{panel_idx+1}.png"
                panel["image_path"] = str(panel_path)
                
                # Generate dialogue if requested
                if args.dialogue:
                    dialogue = generate_dialogue(story, page_idx, panel_idx, args.style)
                    panel["dialogue"] = dialogue
                    print(f"    Dialogue: {dialogue}")
                    
                    # In a real implementation:
                    # if dialogue and os.path.exists(panel_path):
                    #     img = Image.open(panel_path)
                    #     img_with_text = add_text_bubble(img, dialogue, (img.width//2, img.height//2))
                    #     img_with_text.save(panel_path)
            
            except Exception as e:
                print(f"Error generating panel: {e}")
        
        # In a real implementation, create comic page from panels:
        # panel_images = [Image.open(panel["image_path"]) for panel in panels if panel["image_path"]]
        # if panel_images:
        #     comic_page = create_comic_page(panel_images)
        #     comic_page.save(page_dir / "page_layout.png")
        
        # Update story outline with results
        with open(output_dir / "story_outline.json", 'w') as f:
            json.dump(story, f, indent=2)
    
    print(f"\nComic generation complete. Files saved to {output_dir}")
    print(f"Story outline saved to {output_dir / 'story_outline.json'}")

if __name__ == "__main__":
    main()