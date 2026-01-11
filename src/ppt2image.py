from pptxtoimages.tools import PPTXToImageConverter

# Initialize converter
converter = PPTXToImageConverter(pptx_path)

# Convert your .pptx file to images
images = converter.convert("path/to/presentation.pptx", output_dir="output_images")

print(f"Converted {len(images)} slides to images.")