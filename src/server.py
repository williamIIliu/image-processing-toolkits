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

from typing import Any, List, Dict, Tuple, Optional
import asyncio
import base64
import io
import os
import threading
import time
import psutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
import cv2
from functools import lru_cache
import concurrent.futures
from collections import defaultdict
import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

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

# Server initialization
server = Server("image_processor")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available image processing tools"""
    return [
        types.Tool(
            name="crop-image",
            description="Crop image to specified region",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "x": {
                        "type": "integer",
                        "description": "X coordinate of the top-left corner of crop region",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate of the top-left corner of crop region",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Width of the crop region",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Height of the crop region",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "x", "y", "width", "height"],
            },
        ),
        types.Tool(
            name="rotate-image",
            description="Rotate image by specified angle",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees",
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "Whether to expand canvas to fit rotated image",
                        "default": True,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "angle"],
            },
        ),
        types.Tool(
            name="resize-image",
            description="Resize image to specified dimensions",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "width": {
                        "type": "integer",
                        "description": "New width",
                    },
                    "height": {
                        "type": "integer",
                        "description": "New height",
                    },
                    "maintain_aspect": {
                        "type": "boolean",
                        "description": "Whether to maintain aspect ratio",
                        "default": True,
                    },
                    "resample": {
                        "type": "string",
                        "description": "Resampling method",
                        "enum": ["LANCZOS", "BILINEAR", "BICUBIC", "NEAREST"],
                        "default": "LANCZOS",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "width", "height"],
            },
        ),
        types.Tool(
            name="adjust-contrast",
            description="Adjust image contrast",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "factor": {
                        "type": "number",
                        "description": "Contrast adjustment factor (1.0 = original, >1 = enhance, <1 = reduce)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "factor"],
            },
        ),
        types.Tool(
            name="adjust-brightness",
            description="Adjust image brightness",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "factor": {
                        "type": "number",
                        "description": "Brightness adjustment factor (1.0 = original, >1 = brighter, <1 = darker)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "factor"],
            },
        ),
        types.Tool(
            name="adjust-saturation",
            description="Adjust image saturation",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "factor": {
                        "type": "number",
                        "description": "Saturation adjustment factor (1.0 = original, >1 = enhance, <1 = reduce, 0 = grayscale)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "factor"],
            },
        ),
        types.Tool(
            name="apply-filter",
            description="Apply image filter",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "filter_type": {
                        "type": "string",
                        "description": "Filter type",
                        "enum": ["BLUR", "DETAIL", "EDGE_ENHANCE", "EDGE_ENHANCE_MORE", 
                                "EMBOSS", "FIND_EDGES", "SHARPEN", "SMOOTH", "SMOOTH_MORE"],
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "filter_type"],
            },
        ),
        types.Tool(
            name="flip-image",
            description="Flip image horizontally or vertically",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Flip direction",
                        "enum": ["horizontal", "vertical"],
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "direction"],
            },
        ),
        types.Tool(
            name="convert-format",
            description="Convert image format",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "format": {
                        "type": "string",
                        "description": "Target format",
                        "enum": ["JPEG", "PNG", "BMP", "TIFF", "WEBP"],
                    },
                    "quality": {
                        "type": "integer",
                        "description": "JPEG quality (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 95,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "format"],
            },
        ),
        types.Tool(
            name="get-image-info",
            description="Get image information and metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="create-thumbnail",
            description="Create image thumbnail",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "size": {
                        "type": "integer",
                        "description": "Maximum thumbnail size",
                        "default": 128,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="add-watermark",
            description="Add watermark to image",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "watermark_text": {
                        "type": "string",
                        "description": "Watermark text",
                    },
                    "position": {
                        "type": "string",
                        "description": "Watermark position",
                        "enum": ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                        "default": "bottom-right",
                    },
                    "opacity": {
                        "type": "number",
                        "description": "Watermark opacity (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "Font size",
                        "default": 36,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)",
                    },
                },
                "required": ["file_path", "watermark_text"],
            },
        ),
    ]

# Image processing functions
async def crop_image(file_path: str, x: int, y: int, width: int, height: int, output_path: str = None) -> str:
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

async def rotate_image(file_path: str, angle: float, expand: bool = True, output_path: str = None) -> str:
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

async def resize_image(file_path: str, width: int, height: int, maintain_aspect: bool = True, 
                      resample: str = "LANCZOS", output_path: str = None) -> str:
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

async def adjust_contrast(file_path: str, factor: float, output_path: str = None) -> str:
    """Adjust image contrast"""
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

async def adjust_brightness(file_path: str, factor: float, output_path: str = None) -> str:
    """Adjust image brightness"""
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

async def adjust_saturation(file_path: str, factor: float, output_path: str = None) -> str:
    """Adjust image saturation"""
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

async def apply_filter(file_path: str, filter_type: str, output_path: str = None) -> str:
    """Apply image filter"""
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

async def flip_image(file_path: str, direction: str, output_path: str = None) -> str:
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

async def convert_format(file_path: str, format: str, quality: int = 95, output_path: str = None) -> str:
    """Convert image format"""
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

async def get_image_info(file_path: str) -> Dict[str, Any]:
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
        
        return info
        
    except Exception as e:
        return {"Error": f"Error getting image info: {str(e)}"}

async def create_thumbnail(file_path: str, size: int = 128, output_path: str = None) -> str:
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

async def add_watermark(file_path: str, watermark_text: str, position: str = "bottom-right", 
                       opacity: float = 0.5, font_size: int = 36, output_path: str = None) -> str:
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

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool call requests"""
    if not arguments:
        raise ValueError("Missing arguments")

    file_path = arguments.get("file_path")
    if not file_path:
        raise ValueError("Missing file path")

    try:
        if name == "crop-image":
            x = arguments.get("x")
            y = arguments.get("y")
            width = arguments.get("width")
            height = arguments.get("height")
            output_path = arguments.get("output_path")
            result = await crop_image(file_path, x, y, width, height, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "rotate-image":
            angle = arguments.get("angle")
            expand = arguments.get("expand", True)
            output_path = arguments.get("output_path")
            result = await rotate_image(file_path, angle, expand, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "resize-image":
            width = arguments.get("width")
            height = arguments.get("height")
            maintain_aspect = arguments.get("maintain_aspect", True)
            resample = arguments.get("resample", "LANCZOS")
            output_path = arguments.get("output_path")
            result = await resize_image(file_path, width, height, maintain_aspect, resample, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "adjust-contrast":
            factor = arguments.get("factor")
            output_path = arguments.get("output_path")
            result = await adjust_contrast(file_path, factor, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "adjust-brightness":
            factor = arguments.get("factor")
            output_path = arguments.get("output_path")
            result = await adjust_brightness(file_path, factor, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "adjust-saturation":
            factor = arguments.get("factor")
            output_path = arguments.get("output_path")
            result = await adjust_saturation(file_path, factor, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "apply-filter":
            filter_type = arguments.get("filter_type")
            output_path = arguments.get("output_path")
            result = await apply_filter(file_path, filter_type, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "flip-image":
            direction = arguments.get("direction")
            output_path = arguments.get("output_path")
            result = await flip_image(file_path, direction, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "convert-format":
            format = arguments.get("format")
            quality = arguments.get("quality", 95)
            output_path = arguments.get("output_path")
            result = await convert_format(file_path, format, quality, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "get-image-info":
            info = await get_image_info(file_path)
            if "Error" in info:
                return [types.TextContent(type="text", text=info["Error"])]
            
            text = "Image Information:\n"
            for key, value in info.items():
                if key == "EXIF Data" and isinstance(value, dict):
                    text += f"\n{key}:\n"
                    for exif_key, exif_value in value.items():
                        text += f"  {exif_key}: {exif_value}\n"
                else:
                    text += f"{key}: {value}\n"
            
            return [types.TextContent(type="text", text=text)]
        
        elif name == "create-thumbnail":
            size = arguments.get("size", 128)
            output_path = arguments.get("output_path")
            result = await create_thumbnail(file_path, size, output_path)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "add-watermark":
            watermark_text = arguments.get("watermark_text")
            position = arguments.get("position", "bottom-right")
            opacity = arguments.get("opacity", 0.5)
            font_size = arguments.get("font_size", 36)
            output_path = arguments.get("output_path")
            result = await add_watermark(file_path, watermark_text, position, opacity, font_size, output_path)
            return [types.TextContent(type="text", text=result)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except ImageProcessorError as e:
        return [types.TextContent(type="text", text=f"Image processing error: {str(e)}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error processing request: {str(e)}")]

async def main():
    """Run the server"""
    try:
        print("Image Processing MCP Server starting...")
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="image_processor",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        print(f"Server runtime error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
