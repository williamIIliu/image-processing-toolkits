#!/usr/bin/env python3
"""
Image Processing MCP Server

This server provides comprehensive image processing capabilities including:
- Basic operations: crop, rotate, resize, flip
- Color adjustments: brightness, contrast, saturation
- Filters and effects
- Format conversion
- Metadata extraction
- Thumbnail generation
- Watermark addition

The server uses PIL (Pillow) for image processing and includes memory management
and caching for efficient operation.
"""

from typing import Any, Dict, Optional
import os
import threading
import time
import psutil
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
from mcp.server.fastmcp import FastMCP

class ImageProcessorError(Exception):
    """Base class for image processing related errors"""
    pass

class ImageLoadError(ImageProcessorError):
    """Image loading error"""
    pass

class ImageProcessingError(ImageProcessorError):
    """Image processing error"""
    pass

class ImageManager:
    """Image processing manager with caching and memory management"""
    _instance = None
    _lock = threading.Lock()
    _cache = {}
    _last_used = {}
    _memory_threshold = 0.8  # Memory usage threshold
    _max_cache_size = 50  # Maximum number of cached images
    _max_idle_time = 300  # Maximum cache idle time (seconds)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_cache, daemon=True)
        self._cleanup_thread.start()

    def load_image(self, file_path: str) -> Image.Image:
        """Load image with caching support"""
        with self._lock:
            # Check cache
            if file_path in self._cache:
                self._last_used[file_path] = time.time()
                return self._cache[file_path].copy()
            
            # Check memory usage
            self._check_memory_usage()
            
            try:
                # Load image
                image = Image.open(file_path)
                # Convert to RGB mode for compatibility
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Cache image
                if len(self._cache) < self._max_cache_size:
                    self._cache[file_path] = image.copy()
                    self._last_used[file_path] = time.time()
                
                return image
            except Exception as e:
                raise ImageLoadError(f"Unable to load image {file_path}: {str(e)}")

    def _check_memory_usage(self):
        """Check memory usage and clean cache if necessary"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self._memory_threshold:
            self._clear_oldest_cache()

    def _clear_oldest_cache(self):
        """Clear the oldest cache entry"""
        if not self._last_used:
            return
        
        oldest_key = min(self._last_used.items(), key=lambda x: x[1])[0]
        if oldest_key in self._cache:
            del self._cache[oldest_key]
            del self._last_used[oldest_key]

    def _cleanup_cache(self):
        """Periodically clean expired cache"""
        while True:
            time.sleep(60)  # Check every minute
            with self._lock:
                current_time = time.time()
                expired_keys = [
                    key for key, last_used in self._last_used.items()
                    if current_time - last_used > self._max_idle_time
                ]
                for key in expired_keys:
                    if key in self._cache:
                        del self._cache[key]
                    del self._last_used[key]

    def clear_cache(self):
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._last_used.clear()

# Create global ImageManager instance
image_manager = ImageManager()

# Create FastMCP server
mcp = FastMCP("Image Processing Toolkits")

@mcp.tool()
def crop_image(file_path: str, x: int, y: int, width: int, height: int, output_path: Optional[str] = None) -> str:
    """Crop image to specified region"""
    try:
        image = image_manager.load_image(file_path)
        
        # Validate crop region
        img_width, img_height = image.size
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            return f"Crop region exceeds image boundaries. Image size: {img_width}x{img_height}"
        
        # Crop image
        cropped = image.crop((x, y, x + width, y + height))
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_cropped{ext}"
        
        cropped.save(output_path)
        return f"Image cropped and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error cropping image: {str(e)}")

@mcp.tool()
def rotate_image(file_path: str, angle: float, expand: bool = True, output_path: Optional[str] = None) -> str:
    """Rotate image by specified angle"""
    try:
        image = image_manager.load_image(file_path)
        
        # Rotate image
        rotated = image.rotate(angle, expand=expand, fillcolor='white')
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_rotated_{angle}deg{ext}"
        
        rotated.save(output_path)
        return f"Image rotated {angle} degrees and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error rotating image: {str(e)}")

@mcp.tool()
def resize_image(file_path: str, width: int, height: int, maintain_aspect: bool = True, 
                resample: str = "LANCZOS", output_path: Optional[str] = None) -> str:
    """Resize image to specified dimensions"""
    try:
        image = image_manager.load_image(file_path)
        
        # Get resampling method
        resample_methods = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "NEAREST": Image.Resampling.NEAREST,
        }
        resample_method = resample_methods.get(resample, Image.Resampling.LANCZOS)
        
        if maintain_aspect:
            # Maintain aspect ratio
            image.thumbnail((width, height), resample_method)
            resized = image
        else:
            # Force resize to specified dimensions
            resized = image.resize((width, height), resample_method)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_resized_{width}x{height}{ext}"
        
        resized.save(output_path)
        return f"Image resized and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error resizing image: {str(e)}")

@mcp.tool()
def adjust_contrast(file_path: str, factor: float, output_path: Optional[str] = None) -> str:
    """Adjust image contrast (1.0 = original, >1 = enhance, <1 = reduce)"""
    try:
        image = image_manager.load_image(file_path)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_contrast_{factor}{ext}"
        
        enhanced.save(output_path)
        return f"Image contrast adjusted and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error adjusting contrast: {str(e)}")

@mcp.tool()
def adjust_brightness(file_path: str, factor: float, output_path: Optional[str] = None) -> str:
    """Adjust image brightness (1.0 = original, >1 = brighter, <1 = darker)"""
    try:
        image = image_manager.load_image(file_path)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_brightness_{factor}{ext}"
        
        enhanced.save(output_path)
        return f"Image brightness adjusted and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error adjusting brightness: {str(e)}")

@mcp.tool()
def adjust_saturation(file_path: str, factor: float, output_path: Optional[str] = None) -> str:
    """Adjust image saturation (1.0 = original, >1 = enhance, <1 = reduce, 0 = grayscale)"""
    try:
        image = image_manager.load_image(file_path)
        
        # Adjust saturation
        enhancer = ImageEnhance.Color(image)
        enhanced = enhancer.enhance(factor)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_saturation_{factor}{ext}"
        
        enhanced.save(output_path)
        return f"Image saturation adjusted and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error adjusting saturation: {str(e)}")

@mcp.tool()
def apply_filter(file_path: str, filter_type: str, output_path: Optional[str] = None) -> str:
    """Apply image filter (BLUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SHARPEN, SMOOTH, SMOOTH_MORE)"""
    try:
        image = image_manager.load_image(file_path)
        
        # Get filter
        filters = {
            "BLUR": ImageFilter.BLUR,
            "DETAIL": ImageFilter.DETAIL,
            "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
            "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
            "EMBOSS": ImageFilter.EMBOSS,
            "FIND_EDGES": ImageFilter.FIND_EDGES,
            "SHARPEN": ImageFilter.SHARPEN,
            "SMOOTH": ImageFilter.SMOOTH,
            "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
        }
        
        filter_obj = filters.get(filter_type)
        if not filter_obj:
            return f"Unknown filter type: {filter_type}"
        
        # Apply filter
        filtered = image.filter(filter_obj)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_{filter_type.lower()}{ext}"
        
        filtered.save(output_path)
        return f"Applied {filter_type} filter and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error applying filter: {str(e)}")

@mcp.tool()
def flip_image(file_path: str, direction: str, output_path: Optional[str] = None) -> str:
    """Flip image horizontally or vertically"""
    try:
        image = image_manager.load_image(file_path)
        
        # Flip image
        if direction == "horizontal":
            flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            return f"Unknown flip direction: {direction}"
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_flip_{direction}{ext}"
        
        flipped.save(output_path)
        return f"Image flipped {direction} and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error flipping image: {str(e)}")

@mcp.tool()
def convert_format(file_path: str, format: str, quality: int = 95, output_path: Optional[str] = None) -> str:
    """Convert image format (JPEG, PNG, BMP, TIFF, WEBP)"""
    try:
        image = image_manager.load_image(file_path)
        
        # Generate output path
        if not output_path:
            name = os.path.splitext(file_path)[0]
            ext = format.lower()
            if ext == "jpeg":
                ext = "jpg"
            output_path = f"{name}.{ext}"
        
        # Save in new format
        save_kwargs = {}
        if format.upper() == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        
        image.save(output_path, format=format.upper(), **save_kwargs)
        return f"Image converted to {format} format and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error converting format: {str(e)}")

@mcp.tool()
def get_image_info(file_path: str) -> str:
    """Get image information and metadata"""
    try:
        image = image_manager.load_image(file_path)
        
        # Basic information
        info = {
            "File Path": file_path,
            "Format": image.format,
            "Mode": image.mode,
            "Size": f"{image.size[0]}x{image.size[1]}",
            "Width": image.size[0],
            "Height": image.size[1],
        }
        
        # File size
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            info["File Size"] = f"{file_size / 1024:.2f} KB"
        
        # EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            exif_data = {}
            for tag_id, value in image._getexif().items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
            info["EXIF Data"] = exif_data
        
        # Format output
        text = "Image Information:\n"
        for key, value in info.items():
            if key == "EXIF Data" and isinstance(value, dict):
                text += f"\n{key}:\n"
                for exif_key, exif_value in value.items():
                    text += f"  {exif_key}: {exif_value}\n"
            else:
                text += f"{key}: {value}\n"
        
        return text
        
    except Exception as e:
        return f"Error getting image info: {str(e)}"

@mcp.tool()
def create_thumbnail(file_path: str, size: int = 128, output_path: Optional[str] = None) -> str:
    """Create image thumbnail"""
    try:
        image = image_manager.load_image(file_path)
        
        # Create thumbnail
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_thumbnail_{size}{ext}"
        
        image.save(output_path)
        return f"Thumbnail created and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error creating thumbnail: {str(e)}")

@mcp.tool()
def add_watermark(file_path: str, watermark_text: str, position: str = "bottom-right", 
                 opacity: float = 0.5, font_size: int = 36, output_path: Optional[str] = None) -> str:
    """Add watermark to image"""
    try:
        image = image_manager.load_image(file_path)
        
        # Create watermark layer
        watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Try to load font
        try:
            # Try to use system font on Windows
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Try other common fonts
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            except:
                # Use default font
                font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        margin = 20
        positions = {
            "top-left": (margin, margin),
            "top-right": (image.size[0] - text_width - margin, margin),
            "bottom-left": (margin, image.size[1] - text_height - margin),
            "bottom-right": (image.size[0] - text_width - margin, image.size[1] - text_height - margin),
            "center": ((image.size[0] - text_width) // 2, (image.size[1] - text_height) // 2),
        }
        
        pos = positions.get(position, positions["bottom-right"])
        
        # Draw watermark
        alpha = int(255 * opacity)
        draw.text(pos, watermark_text, font=font, fill=(255, 255, 255, alpha))
        
        # Merge images
        watermarked = Image.alpha_composite(image.convert('RGBA'), watermark)
        watermarked = watermarked.convert('RGB')
        
        # Save result
        if not output_path:
            name, ext = os.path.splitext(file_path)
            output_path = f"{name}_watermarked{ext}"
        
        watermarked.save(output_path)
        return f"Watermark added and saved to: {output_path}"
        
    except Exception as e:
        raise ImageProcessingError(f"Error adding watermark: {str(e)}")

if __name__ == "__main__":
    mcp.run()
