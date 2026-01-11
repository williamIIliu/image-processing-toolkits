#!/usr/bin/env python3
"""Test script for watermark functionality."""

import os
import sys
from PIL import Image, ImageDraw, ImageFont


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


def add_watermark(file_path: str, watermark_text: str, position: str = "bottom-right",
                 opacity: float = 0.5, font_size: int = 36, output_path: str = None) -> str:
    """Add a semi-transparent text watermark to an image.

    The watermark is drawn on a transparent overlay and then composited with
    the original image. This is typically used for branding or copyright
    marking.

    Args:
        file_path: Path to the source image file.
        watermark_text: Text content of the watermark.
        position: Watermark position. One of "top-left", "top-right",
            "bottom-left", "bottom-right", "center".
        opacity: Watermark opacity in the range [0.0, 1.0].
        font_size: Font size of the watermark text.
        output_path: Optional path for the output image. If not provided,
            a file named "<original>_watermarked<ext>" will be created.

    Returns:
        A human-readable message describing where the watermarked image is
        saved.
    """
    try:
        image = Image.open(file_path)
        
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


def create_test_image(output_path: str, size: tuple = (800, 600), color: str = "#4A90E2"):
    """Create a simple test image."""
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    
    # Add some text to make it more realistic
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Test Image", font=font, fill=(255, 255, 255))
    draw.text((50, 100), f"Size: {size[0]}x{size[1]}", font=font, fill=(255, 255, 255))
    
    image.save(output_path)
    print(f"✓ Created test image: {output_path}")
    return output_path


def test_watermark_basic():
    """Test basic watermark functionality."""
    print("\n=== Test 1: Basic Watermark ===")
    test_image = "test_image_basic.jpg"
    create_test_image(test_image)
    
    try:
        result = add_watermark(
            file_path=test_image,
            watermark_text="Test Watermark",
            position="bottom-right",
            opacity=0.6,
            font_size=36
        )
        print(f"✓ {result}")
        
        # Verify output file exists
        expected_output = "test_image_basic_watermarked.jpg"
        if os.path.exists(expected_output):
            print(f"✓ Output file created: {expected_output}")
            img = Image.open(expected_output)
            print(f"✓ Output image size: {img.size}")
            return True
        else:
            print(f"✗ Output file not found: {expected_output}")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_watermark_positions():
    """Test watermark at different positions."""
    print("\n=== Test 2: Different Positions ===")
    test_image = "test_image_positions.jpg"
    create_test_image(test_image)
    
    positions = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
    success_count = 0
    
    for pos in positions:
        try:
            output_path = f"test_image_positions_{pos.replace('-', '_')}_watermarked.jpg"
            result = add_watermark(
                file_path=test_image,
                watermark_text=f"Watermark at {pos}",
                position=pos,
                opacity=0.5,
                font_size=24,
                output_path=output_path
            )
            print(f"✓ {pos}: {result}")
            
            if os.path.exists(output_path):
                success_count += 1
            else:
                print(f"✗ {pos}: Output file not created")
        except Exception as e:
            print(f"✗ {pos}: {e}")
    
    # Cleanup
    for f in [test_image] + [f"test_image_positions_{p.replace('-', '_')}_watermarked.jpg" for p in positions]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"✓ Passed {success_count}/{len(positions)} position tests")
    return success_count == len(positions)


def test_watermark_opacity():
    """Test watermark with different opacity levels."""
    print("\n=== Test 3: Different Opacity Levels ===")
    test_image = "test_image_opacity.jpg"
    create_test_image(test_image)
    
    opacities = [0.2, 0.5, 0.8, 1.0]
    success_count = 0
    
    for opacity in opacities:
        try:
            output_path = f"test_image_opacity_{int(opacity*100)}_watermarked.jpg"
            result = add_watermark(
                file_path=test_image,
                watermark_text=f"Opacity {opacity}",
                position="center",
                opacity=opacity,
                font_size=30,
                output_path=output_path
            )
            print(f"✓ Opacity {opacity}: {result}")
            
            if os.path.exists(output_path):
                success_count += 1
            else:
                print(f"✗ Opacity {opacity}: Output file not created")
        except Exception as e:
            print(f"✗ Opacity {opacity}: {e}")
    
    # Cleanup
    for f in [test_image] + [f"test_image_opacity_{int(o*100)}_watermarked.jpg" for o in opacities]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"✓ Passed {success_count}/{len(opacities)} opacity tests")
    return success_count == len(opacities)


def test_watermark_font_sizes():
    """Test watermark with different font sizes."""
    print("\n=== Test 4: Different Font Sizes ===")
    test_image = "test_image_fonts.jpg"
    create_test_image(test_image)
    
    font_sizes = [24, 48, 72, 96]
    success_count = 0
    
    for font_size in font_sizes:
        try:
            output_path = f"test_image_fonts_{font_size}_watermarked.jpg"
            result = add_watermark(
                file_path=test_image,
                watermark_text=f"Font Size {font_size}",
                position="center",
                opacity=0.7,
                font_size=font_size,
                output_path=output_path
            )
            print(f"✓ Font size {font_size}: {result}")
            
            if os.path.exists(output_path):
                success_count += 1
            else:
                print(f"✗ Font size {font_size}: Output file not created")
        except Exception as e:
            print(f"✗ Font size {font_size}: {e}")
    
    # Cleanup
    for f in [test_image] + [f"test_image_fonts_{fs}_watermarked.jpg" for fs in font_sizes]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"✓ Passed {success_count}/{len(font_sizes)} font size tests")
    return success_count == len(font_sizes)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Watermark Functionality Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Basic Watermark", test_watermark_basic()))
    results.append(("Different Positions", test_watermark_positions()))
    results.append(("Different Opacity", test_watermark_opacity()))
    results.append(("Different Font Sizes", test_watermark_font_sizes()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Watermark functionality is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())