import imageio
import os
import numpy as np
from PIL import Image

# Get all frame files
frames = sorted([f for f in os.listdir('Music_Video') if f.startswith('f') and f.endswith('.png')])

target_height = 1088  # 1080 rounded up to nearest multiple of 16
target_width = 1920   # already a multiple of 16

for frame_file in frames:
    frame_path = os.path.join('Music_Video', frame_file)
    img = Image.open(frame_path)
    # Resize to target dimensions
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    img_resized.save(frame_path)
    print(f'Resized {frame_file} to {target_height}x{target_width}')

print(f'Done! Resized {len(frames)} frames.')
