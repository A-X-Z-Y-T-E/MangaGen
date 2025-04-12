"""
MangaGen Web UI - Streamlit interface for the manga generator.
Run with: streamlit run ui/web_ui.py
"""

import os
import sys
import streamlit as st
import torch
import io
from PIL import Image
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import manga generation modules
from manga_prompt_generator import MangaPromptGenerator
from enhanced_manga_generator import parse_args as parse_enhanced_args
from generate_manga_story import parse_args as parse_story_args
from utils.image_utils import apply_manga_style, enhance_manga_quality, add_manga_effects, create_manga_panel_border, add_speed_lines, add_manga_tones

# Import modified main functions
from simple_manga_generator import main as simple_main
from enhanced_manga_generator import main as enhanced_main
from generate_manga_story import main as story_main

# Set page config
st.set_page_config(
    page_title="MangaGen",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("MangaGen: AI-Powered Manga Generator")
st.markdown("Create manga-style images and comics from text prompts using AI.")

# Sidebar for generation mode
st.sidebar.title("Generation Mode")
generation_mode = st.sidebar.radio(
    "Select Mode",
    ["Simple Panel", "Enhanced Panel", "Multi-Panel Story"]
)

# Sidebar for common parameters
st.sidebar.title("Common Parameters")

prompt = st.sidebar.text_area("Prompt", "A samurai facing a dragon in battle", help="Text description of what you want to generate")
seed = st.sidebar.number_input("Seed", min_value=0, max_value=2147483647, step=1, value=None, help="Random seed for reproducibility (blank for random)")
if seed == 0:
    seed = None

# Advanced options (collapsible)
with st.sidebar.expander("Advanced Options"):
    steps = st.number_input("Denoising Steps", min_value=10, max_value=100, value=30)
    guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)
    model_version = st.selectbox(
        "TrinArt Model Version",
        ["diffusers-60k", "diffusers-95k", "diffusers-115k"],
        index=2,
        help="Model version: 60k for lighter stylization, 115k for stronger"
    )

# Parameters specific to each mode
if generation_mode == "Simple Panel":
    st.subheader("Simple Manga Panel Generator")
    st.markdown("Generate a basic manga panel with minimal processing.")
    
    if st.button("Generate Simple Panel"):
        with st.spinner("Generating manga panel..."):
            args = parse_enhanced_args([
                "--prompt", prompt,
                "--output", "output/simple_output.png",
                "--steps", str(steps),
                "--guidance_scale", str(guidance_scale),
                "--width", str(width),
                "--height", str(height),
                "--model_version", model_version
            ])
            
            if seed is not None:
                args.seed = seed
                
            # Run generation
            try:
                simple_main(args)
                st.image("output/simple_output.png", caption="Generated Manga Panel", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")

elif generation_mode == "Enhanced Panel":
    st.subheader("Enhanced Manga Panel Generator")
    st.markdown("Generate a manga panel with advanced styling and effects.")
    
    # Enhanced mode options
    col1, col2 = st.columns(2)
    
    with col1:
        manga_style = st.selectbox(
            "Manga Style",
            ["standard", "highcontrast", "sketch", "screentone", "action"],
            index=0
        )
        enhance = st.checkbox("Enhance Quality", value=True)
        border = st.checkbox("Add Panel Border", value=True)
    
    with col2:
        effect = st.selectbox(
            "Special Effect",
            [None, "action", "emotional", "impact", "background"],
            index=0
        )
        speed_lines = st.checkbox("Add Speed Lines", value=False)
        speed_direction = st.selectbox(
            "Speed Lines Direction",
            ["radial", "horizontal", "vertical"],
            index=0,
            disabled=not speed_lines
        )
    
    if st.button("Generate Enhanced Panel"):
        with st.spinner("Generating enhanced manga panel..."):
            cmd_args = [
                "--prompt", prompt,
                "--output", "output/enhanced_output.png",
                "--steps", str(steps),
                "--guidance_scale", str(guidance_scale),
                "--width", str(width),
                "--height", str(height),
                "--model_version", model_version,
                "--style", manga_style
            ]
            
            if seed is not None:
                cmd_args.extend(["--seed", str(seed)])
            if enhance:
                cmd_args.append("--enhance")
            if border:
                cmd_args.append("--border")
            if effect:
                cmd_args.extend(["--effect", effect])
            if speed_lines:
                cmd_args.append("--speed_lines")
                cmd_args.extend(["--speed_direction", speed_direction])
                
            args = parse_enhanced_args(cmd_args)
            
            try:
                enhanced_main(args)
                st.image("output/enhanced_output.png", caption="Enhanced Manga Panel", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")

elif generation_mode == "Multi-Panel Story":
    st.subheader("Multi-Panel Manga Story Generator")
    st.markdown("Generate a complete manga story with multiple panels.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_panels = st.slider("Number of Panels", min_value=2, max_value=9, value=4)
        manga_style = st.selectbox(
            "Visual Style",
            ["highcontrast", "sketch", "screentone", "action"],
            index=0
        )
        style = st.selectbox(
            "Manga Genre",
            ["manga", "shonen", "seinen", "shoujo", "action", "noir"],
            index=0
        )
    
    with col2:
        layout = st.selectbox(
            "Panel Layout",
            ["vertical", "horizontal", "grid"],
            index=0
        )
        effect = st.selectbox(
            "Special Effects",
            [None, "action", "emotional", "impact", "background"],
            index=0
        )
        
    # Additional options
    enhance = st.checkbox("Enhance Quality", value=True)
    tones = st.checkbox("Add Manga Tones", value=False)
    borders = st.checkbox("Add Panel Borders", value=True)
    save_individual = st.checkbox("Save Individual Panels", value=True)
    
    # UI for Groq API key (if needed)
    groq_api_key = st.text_input("Groq API Key (optional)", 
                                type="password", 
                                help="Enter your Groq API key if not set in .env file")
    
    if st.button("Generate Manga Story"):
        with st.spinner(f"Generating {num_panels}-panel manga story..."):
            # Set API key if provided
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                
            cmd_args = [
                "--prompt", prompt,
                "--panels", str(num_panels),
                "--style", style,
                "--layout", layout,
                "--steps", str(steps),
                "--guidance_scale", str(guidance_scale),
                "--width", str(width),
                "--height", str(height),
                "--diffusion_model", model_version,
                "--manga_style", manga_style,
                "--output_dir", "output"
            ]
            
            if seed is not None:
                cmd_args.extend(["--seed", str(seed)])
            if enhance:
                cmd_args.append("--enhance")
            if effect:
                cmd_args.extend(["--effects", effect])
            if tones:
                cmd_args.append("--tones")
            if borders:
                cmd_args.append("--borders")
            if save_individual:
                cmd_args.append("--save_individual")
                
            args = parse_story_args(cmd_args)
            
            try:
                story_main(args)
                
                # Display the result
                layout_path = f"output/manga_story_{style}.png"
                st.image(layout_path, caption=f"Generated {num_panels}-Panel Manga Story", use_column_width=True)
                
                # Display individual panels if saved
                if save_individual:
                    st.subheader("Individual Panels")
                    panel_cols = st.columns(min(num_panels, 4))
                    for i in range(num_panels):
                        panel_path = f"output/panel_{i+1}.png"
                        if os.path.exists(panel_path):
                            panel_cols[i % 4].image(panel_path, caption=f"Panel {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating manga story: {str(e)}")

# Footer
st.markdown("---")
st.markdown("MangaGen - Powered by TrinArt Stable Diffusion and Groq LLM")