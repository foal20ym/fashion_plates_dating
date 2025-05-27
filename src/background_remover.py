from rembg import remove
from PIL import Image
import os
import io

def process_image(input_path, output_path):
    # Read and remove background
    with open(input_path, 'rb') as inp_file:
        input_data = inp_file.read()
        output_data = remove(input_data)

    # Open result as RGBA
    img = Image.open(io.BytesIO(output_data)).convert("RGBA")

    # Convert transparent to black
    black_bg = Image.new("RGB", img.size, (0, 0, 0))
    black_bg.paste(img, mask=img.split()[3])  # Apply alpha channel as mask

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    black_bg.save(output_path)

def process_all_images(input_root, output_root, exclude_dirs=None):
    # Normalize the exclude_dirs paths for consistent comparison
    normalized_exclude_dirs = []
    if exclude_dirs:
        normalized_exclude_dirs = [os.path.normpath(dir_path) for dir_path in exclude_dirs]
    
    for dirpath, _, filenames in os.walk(input_root):
        # Skip the excluded directories
        norm_path = os.path.normpath(dirpath)
        if any(norm_path.startswith(excluded) for excluded in normalized_exclude_dirs):
            print(f"Excluding: {dirpath}")
            continue
            
        for fname in filenames:
            if fname.lower().endswith('.png'): # BARA PNG JUST NU
                input_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                process_image(input_path, output_path)


print("Starting background removal process")
private_exclude_dirs = [
    "data/datasets/private/1840/",
    "data/datasets/private/1850/",
    "data/datasets/private/1860/"
]

public_exclude_dirs = [
    "data/datasets/public/1830/",
    "data/datasets/public/1840/"
]

process_all_images('data/datasets/private', 'data/datasets_cleaned/private')
process_all_images('data/datasets/public', 'data/datasets_cleaned/public')

print("Background removal process completed")