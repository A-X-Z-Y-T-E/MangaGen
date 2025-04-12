# Functions to save, resize, stylize images

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from pathlib import Path

def save_image(image, output_path, format="PNG", quality=95):
    """
    Save a PIL Image to disk
    
    Args:
        image: PIL Image object
        output_path: Path to save the image
        format: Image format (PNG, JPEG, etc.)
        quality: Quality for JPEG compression (1-100)
        
    Returns:
        str: Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the image
    image.save(output_path, format=format, quality=quality)
    return output_path

def save_images(images, output_dir, prefix="panel", format="png"):
    """
    Save a list of images to disk
    
    Args:
        images: List of PIL Image objects
        output_dir: Directory to save images
        prefix: Filename prefix
        format: Image format (png, jpg)
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, img in enumerate(images):
        filename = f"{prefix}_{i+1:03d}.{format.lower()}"
        filepath = os.path.join(output_dir, filename)
        
        img.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths

def resize_image(image, width=None, height=None, maintain_aspect_ratio=True):
    """
    Resize a PIL Image
    
    Args:
        image: PIL Image object
        width: New width
        height: New height
        maintain_aspect_ratio: Whether to maintain the aspect ratio
        
    Returns:
        PIL Image: Resized image
    """
    original_width, original_height = image.size
    
    if width is None and height is None:
        return image
    
    if maintain_aspect_ratio:
        if width is None:
            ratio = height / original_height
            width = int(original_width * ratio)
        elif height is None:
            ratio = width / original_width
            height = int(original_height * ratio)
        else:
            # Calculate which dimension to fit to maintain aspect ratio
            width_ratio = width / original_width
            height_ratio = height / original_height
            
            if width_ratio < height_ratio:
                height = int(original_height * width_ratio)
            else:
                width = int(original_width * height_ratio)
    
    # Resize image
    resized_image = image.resize((width, height), Image.LANCZOS)
    return resized_image

def apply_manga_style(image, style="highcontrast"):
    """
    Apply a specific manga style to the image.
    
    Args:
        image (PIL.Image): Input image
        style (str): Style to apply - "highcontrast", "sketch", "screentone", "action"
        
    Returns:
        PIL.Image: Styled image
    """
    if style == "highcontrast":
        # High contrast black and white manga style
        image = image.convert("L")  # Convert to grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
    elif style == "sketch":
        # Sketch-like manga style
        image = image.convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)
        image = ImageOps.invert(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
    elif style == "screentone":
        # Apply screentone-like effect
        image = image.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        # Simulate screentone by adding noise
        img_array = np.array(image)
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
    elif style == "action":
        # Action manga style with emphasized lines
        image = image.convert("L")
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
    
    return image

def add_speed_lines(image, intensity=1.0, direction="radial", focus_point=None):
    """
    Add speed lines to the image to create a sense of motion.
    
    Args:
        image (PIL.Image): Input image
        intensity (float): Intensity of the effect (0-1)
        direction (str): Direction of speed lines - "horizontal", "vertical", or "radial"
        focus_point (tuple): Point to focus radial lines (center if None)
        
    Returns:
        PIL.Image: Image with speed lines
    """
    # Create a copy of the image
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    width, height = image.size
    
    # Set focus point for radial lines
    if focus_point is None:
        focus_point = (width // 2, height // 2)
    
    # Determine line parameters based on intensity
    num_lines = int(50 * intensity)
    line_color = 0  # Black
    line_width = max(1, int(2 * intensity))
    
    if direction == "horizontal":
        # Draw horizontal speed lines
        for _ in range(num_lines):
            y = random.randint(0, height)
            line_length = random.randint(int(width * 0.3), int(width * 0.8))
            start_x = random.randint(0, width - line_length)
            draw.line([(start_x, y), (start_x + line_length, y)], fill=line_color, width=line_width)
            
    elif direction == "vertical":
        # Draw vertical speed lines
        for _ in range(num_lines):
            x = random.randint(0, width)
            line_length = random.randint(int(height * 0.3), int(height * 0.8))
            start_y = random.randint(0, height - line_length)
            draw.line([(x, start_y), (x, start_y + line_length)], fill=line_color, width=line_width)
            
    else:  # "radial"
        # Draw radial speed lines from focus point
        for _ in range(num_lines):
            angle = random.uniform(0, 2 * np.pi)
            line_length = random.randint(int(min(width, height) * 0.2), 
                                        int(min(width, height) * 0.5))
            
            # Calculate end point from focus point
            end_x = int(focus_point[0] + np.cos(angle) * line_length)
            end_y = int(focus_point[1] + np.sin(angle) * line_length)
            
            draw.line([focus_point, (end_x, end_y)], fill=line_color, width=line_width)
    
    return result

def create_manga_panel_border(image, border_width=3, color=0):
    """
    Add a manga panel border to the image.
    
    Args:
        image (PIL.Image): Input image
        border_width (int): Width of the border
        color (int): Border color (0 for black)
        
    Returns:
        PIL.Image: Image with panel border
    """
    # Get image dimensions
    width, height = image.size
    
    # Create a new image with the border
    bordered_image = image.copy()
    draw = ImageDraw.Draw(bordered_image)
    
    # Draw rectangle border
    draw.rectangle(
        [(0, 0), (width-1, height-1)],  # -1 to stay within image bounds
        outline=color,
        width=border_width
    )
    
    return bordered_image

def create_comic_page(panels, layout=None, page_width=1200, page_height=1600, margin=10):
    """
    Create a comic page from multiple panel images
    
    Args:
        panels: List of PIL Image objects
        layout: Layout specification (list of (x, y, w, h) normalized coordinates)
        page_width: Width of the page in pixels
        page_height: Height of the page in pixels
        margin: Margin between panels in pixels
    
    Returns:
        PIL Image: Composite comic page
    """
    # Create blank page with white background
    page = Image.new('RGB', (page_width, page_height), color=(255, 255, 255))
    
    # Default layouts based on number of panels
    default_layouts = {
        1: [(0, 0, 1, 1)],  # Full page
        2: [(0, 0, 1, 0.5), (0, 0.5, 1, 0.5)],  # Horizontal split
        3: [(0, 0, 1, 0.33), (0, 0.33, 1, 0.33), (0, 0.66, 1, 0.34)],  # 3 horizontal rows
        4: [(0, 0, 0.5, 0.5), (0.5, 0, 0.5, 0.5), 
            (0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)],  # 2x2 grid
        6: [(0, 0, 0.33, 0.5), (0.33, 0, 0.33, 0.5), (0.66, 0, 0.34, 0.5),
            (0, 0.5, 0.33, 0.5), (0.33, 0.5, 0.33, 0.5), (0.66, 0.5, 0.34, 0.5)]  # 3x2 grid
    }
    
    # Use provided layout or select default based on panel count
    if layout is None:
        if len(panels) in default_layouts:
            layout = default_layouts[len(panels)]
        else:
            # Fallback to grid layout for any other number
            cols = int(np.ceil(np.sqrt(len(panels))))
            rows = int(np.ceil(len(panels) / cols))
            
            cell_width = 1.0 / cols
            cell_height = 1.0 / rows
            
            layout = []
            for i in range(len(panels)):
                row = i // cols
                col = i % cols
                layout.append((col * cell_width, row * cell_height, cell_width, cell_height))
    
    # Place each panel according to layout
    draw = ImageDraw.Draw(page)
    
    for i, (panel, (x, y, w, h)) in enumerate(zip(panels, layout)):
        # Calculate panel position with margins
        x_px = int(x * page_width) + margin
        y_px = int(y * page_height) + margin
        width_px = int(w * page_width) - (2 * margin)
        height_px = int(h * page_height) - (2 * margin)
        
        # Resize panel to fit position
        resized_panel = panel.copy()
        resized_panel = resized_panel.resize((width_px, height_px), Image.LANCZOS)
        
        # Place on page
        page.paste(resized_panel, (x_px, y_px))
        
        # Draw border around panel
        draw.rectangle([(x_px, y_px), (x_px + width_px, y_px + height_px)], 
                      outline=(0, 0, 0), width=2)
    
    return page

def add_text_bubble(image, text, position, bubble_type="speech", font_size=16, padding=10):
    """
    Add a text bubble to an image
    
    Args:
        image: PIL Image object
        text: Text to display in the bubble
        position: (x, y) position for the bubble
        bubble_type: Type of bubble ("speech", "thought", "shout")
        font_size: Font size for the text
        padding: Padding around the text in pixels
        
    Returns:
        PIL Image: Image with text bubble
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a copy of the image
    result = image.copy().convert('RGBA')
    draw = ImageDraw.Draw(result)
    
    # Try to load font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text size
    text_width, text_height = draw.textsize(text, font=font)
    
    # Calculate bubble size
    bubble_width = text_width + 2 * padding
    bubble_height = text_height + 2 * padding
    
    # Calculate bubble position
    x, y = position
    bubble_x = x - bubble_width // 2
    bubble_y = y - bubble_height // 2
    
    # Ensure bubble is within image boundaries
    width, height = image.size
    bubble_x = max(5, min(width - bubble_width - 5, bubble_x))
    bubble_y = max(5, min(height - bubble_height - 5, bubble_y))
    
    # Draw different bubble types
    if bubble_type == "speech":
        # Draw speech bubble (rounded rectangle)
        draw.rounded_rectangle(
            [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
            fill=(255, 255, 255, 220),
            outline=(0, 0, 0, 255),
            width=2,
            radius=10
        )
        
        # Draw speech pointer
        pointer_size = 15
        pointer_x = x
        pointer_y = bubble_y + bubble_height
        draw.polygon(
            [(pointer_x - 10, pointer_y),
             (pointer_x + 10, pointer_y),
             (pointer_x, pointer_y + pointer_size)],
            fill=(255, 255, 255, 220),
            outline=(0, 0, 0, 255)
        )
        
    elif bubble_type == "thought":
        # Draw thought bubble (cloud-like shape)
        draw.ellipse(
            [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
            fill=(255, 255, 255, 220),
            outline=(0, 0, 0, 255),
            width=2
        )
        
        # Draw thought bubbles leading to character
        bubble_sizes = [8, 6, 4]
        last_x, last_y = bubble_x + bubble_width // 2, bubble_y + bubble_height
        for size in bubble_sizes:
            center_x = last_x + (x - last_x) * 0.4
            center_y = last_y + (y - last_y) * 0.4
            draw.ellipse(
                [(center_x - size, center_y - size), 
                 (center_x + size, center_y + size)],
                fill=(255, 255, 255, 220),
                outline=(0, 0, 0, 255),
                width=1
            )
            last_x, last_y = center_x, center_y
            
    elif bubble_type == "shout":
        # Draw shout bubble (jagged rectangle)
        points = []
        steps = 20
        for i in range(steps):
            progress = i / steps
            jag_size = 5 if i % 2 == 0 else 0
            
            if i < steps // 4:  # Top edge
                x_pos = bubble_x + progress * 4 * bubble_width // steps
                y_pos = bubble_y - jag_size
            elif i < steps // 2:  # Right edge
                x_pos = bubble_x + bubble_width + jag_size
                y_pos = bubble_y + (progress - 0.25) * 4 * bubble_height // steps
            elif i < 3 * steps // 4:  # Bottom edge
                x_pos = bubble_x + bubble_width - (progress - 0.5) * 4 * bubble_width // steps
                y_pos = bubble_y + bubble_height + jag_size
            else:  # Left edge
                x_pos = bubble_x - jag_size
                y_pos = bubble_y + bubble_height - (progress - 0.75) * 4 * bubble_height // steps
                
            points.append((x_pos, y_pos))
            
        draw.polygon(
            points,
            fill=(255, 255, 255, 220),
            outline=(0, 0, 0, 255)
        )
    
    # Draw text
    text_x = bubble_x + padding
    text_y = bubble_y + padding
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))
    
    return result.convert('RGB')

def enhance_manga_quality(image, contrast=1.3, sharpness=1.5, brightness=1.1):
    """
    Enhance the quality of a manga image by improving contrast, sharpness, and brightness.
    
    Args:
        image (PIL.Image): Input image
        contrast (float): Contrast enhancement factor
        sharpness (float): Sharpness enhancement factor 
        brightness (float): Brightness enhancement factor
        
    Returns:
        PIL.Image: Enhanced image
    """
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    return image

def add_manga_tones(image, density=0.7, pattern_type="dots"):
    """
    Add manga tone patterns to the image.
    
    Args:
        image (PIL.Image): Input image
        density (float): Density of the tone pattern (0-1)
        pattern_type (str): Type of pattern - "dots", "lines", "crosshatch"
        
    Returns:
        PIL.Image: Image with manga tones
    """
    # Convert to grayscale if not already
    if image.mode != "L":
        image = image.convert("L")
    
    # Create a new image for the tone pattern
    width, height = image.size
    pattern = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(pattern)
    
    # Create different patterns based on type
    if pattern_type == "dots":
        # Create dot pattern
        dot_size = int(5 * (1 - density) + 1)  # Smaller value for higher density
        spacing = int(10 * (1 - density) + 5)  # Smaller value for higher density
        
        for y in range(0, height, spacing):
            offset = 0 if (y // spacing) % 2 == 0 else spacing // 2
            for x in range(offset, width, spacing):
                draw.ellipse([x, y, x + dot_size, y + dot_size], fill=0)
                
    elif pattern_type == "lines":
        # Create line pattern
        line_spacing = int(8 * (1 - density) + 2)  # Smaller value for higher density
        for y in range(0, height, line_spacing):
            draw.line([(0, y), (width, y)], fill=0, width=1)
            
    elif pattern_type == "crosshatch":
        # Create crosshatch pattern
        line_spacing = int(12 * (1 - density) + 4)  # Smaller value for higher density
        # Horizontal lines
        for y in range(0, height, line_spacing):
            draw.line([(0, y), (width, y)], fill=0, width=1)
        # Vertical lines
        for x in range(0, width, line_spacing):
            draw.line([(x, 0), (x, height)], fill=0, width=1)
    
    # Apply the pattern based on the image brightness
    img_array = np.array(image)
    pattern_array = np.array(pattern)
    
    # Create mask where darker areas get more pattern
    threshold = np.int_(255 * density)
    mask = img_array < threshold
    
    # Apply pattern to the mask
    result_array = img_array.copy()
    result_array[mask] = np.minimum(result_array[mask], pattern_array[mask])
    
    return Image.fromarray(result_array)

def add_manga_effects(image, effect_type="action"):
    """
    Add specific manga effects to the image.
    
    Args:
        image (PIL.Image): Input image
        effect_type (str): Type of effect - "action", "emotional", "impact", "background"
        
    Returns:
        PIL.Image: Image with manga effect
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = image.size
    
    if effect_type == "action":
        # Add action lines (similar to speed lines but more intense)
        return add_speed_lines(image, intensity=1.5, direction="radial")
        
    elif effect_type == "emotional":
        # Add emotional effect (soft background with vignette)
        # Create vignette
        img_array = np.array(image)
        
        # Generate a circular mask for vignette
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        center_x, center_y = width // 2, height // 2
        
        # Calculate distance from center for each pixel
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance to 0-1 range
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = dist / max_dist
        
        # Create vignette mask
        vignette = 1 - dist * 0.7  # 0.7 controls vignette intensity
        vignette = np.clip(vignette, 0, 1)
        
        # Apply vignette
        if image.mode == "L":
            result_array = img_array * vignette
        else:
            # Apply to all channels if RGB
            vignette = np.stack([vignette] * 3, axis=2)
            result_array = img_array * vignette
            
        result = Image.fromarray(np.uint8(result_array))
        return result
        
    elif effect_type == "impact":
        # Add impact effect (starburst)
        center = (width // 2, height // 2)
        
        # Draw starburst/impact lines
        num_lines = 20
        for i in range(num_lines):
            angle = i * (2 * np.pi / num_lines)
            line_length = min(width, height) * 0.4
            
            end_x = center[0] + int(np.cos(angle) * line_length)
            end_y = center[1] + int(np.sin(angle) * line_length)
            
            draw.line([center, (end_x, end_y)], fill=0, width=2)
            
        # Add impact circle
        circle_radius = min(width, height) * 0.1
        draw.ellipse([
            center[0] - circle_radius, 
            center[1] - circle_radius,
            center[0] + circle_radius, 
            center[1] + circle_radius
        ], outline=0, width=2)
        
        return result
        
    elif effect_type == "background":
        # Add background effect (hatching or patterns)
        background = Image.new("L", (width, height), 255)
        bg_draw = ImageDraw.Draw(background)
        
        # Create a hatching pattern
        line_spacing = 15
        
        # Draw diagonal lines
        for i in range(-height, width + height, line_spacing):
            bg_draw.line([
                (max(0, i), max(0, -i)), 
                (min(width, i + height), min(height, width - i))
            ], fill=200, width=1)
            
        # Blend with original image
        img_array = np.array(image)
        bg_array = np.array(background)
        
        # Create a mask where foreground is dark
        if image.mode == "L":
            threshold = 100
            mask = img_array < threshold
            
            result_array = img_array.copy()
            # Only apply background to lighter areas
            result_array[~mask] = np.minimum(result_array[~mask], bg_array[~mask])
        else:
            # Convert to grayscale for masking
            gray = np.array(image.convert("L"))
            threshold = 100
            mask = gray < threshold
            
            result_array = img_array.copy()
            for i in range(3):  # For each color channel
                channel = result_array[:,:,i]
                channel[~mask] = np.minimum(channel[~mask], bg_array[~mask])
                result_array[:,:,i] = channel
        
        return Image.fromarray(result_array)
    
    # Default case - return original image if effect type not recognized
    return result

def create_manga_layout(images, layout_type="vertical", margin=5, bg_color=255):
    """
    Create a manga layout from multiple panel images.
    
    Args:
        images (list): List of PIL.Image objects
        layout_type (str): Type of layout - "vertical", "horizontal", "grid"
        margin (int): Margin between images
        bg_color (int): Background color
        
    Returns:
        PIL.Image: Composed layout
    """
    if not images:
        raise ValueError("No images provided")
    
    if layout_type == "vertical":
        # Calculate total height
        width = max(img.width for img in images)
        height = sum(img.height for img in images) + margin * (len(images) - 1)
        
        # Create a new image
        result = Image.new("L", (width, height), bg_color)
        
        # Paste images vertically
        y_offset = 0
        for img in images:
            result.paste(img, ((width - img.width) // 2, y_offset))
            y_offset += img.height + margin
        
    elif layout_type == "horizontal":
        # Calculate total width
        width = sum(img.width for img in images) + margin * (len(images) - 1)
        height = max(img.height for img in images)
        
        # Create a new image
        result = Image.new("L", (width, height), bg_color)
        
        # Paste images horizontally
        x_offset = 0
        for img in images:
            result.paste(img, (x_offset, (height - img.height) // 2))
            x_offset += img.width + margin
            
    elif layout_type == "grid":
        # Try to create a square-ish grid
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # Calculate grid dimensions
        grid_width = min(grid_size, num_images)
        grid_height = int(np.ceil(num_images / grid_width))
        
        # Get max image dimensions
        img_width = max(img.width for img in images)
        img_height = max(img.height for img in images)
        
        # Calculate total dimensions
        width = img_width * grid_width + margin * (grid_width - 1)
        height = img_height * grid_height + margin * (grid_height - 1)
        
        # Create a new image
        result = Image.new("L", (width, height), bg_color)
        
        # Paste images in grid
        for i, img in enumerate(images):
            if i >= num_images:
                break
                
            row = i // grid_width
            col = i % grid_width
            
            x_offset = col * (img_width + margin)
            y_offset = row * (img_height + margin)
            
            # Center the image in its cell
            x_center = x_offset + (img_width - img.width) // 2
            y_center = y_offset + (img_height - img.height) // 2
            
            result.paste(img, (x_center, y_center))
    
    return result
