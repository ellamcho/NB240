import os
import numpy as np
from binvox_rw import read_as_3d_array
import matplotlib.pyplot as plt
from PIL import Image

# Path to folder containing .binvox files
binvox_dir = "/Users/ellacho/Documents/NB240/ShapeNetSem-backup/models-binvox"
output_dir = "/Users/ellacho/Documents/NB240/ShapeNetSem-backup/models_binvox_png"
os.makedirs(output_dir, exist_ok=True)

def binvox_to_image(voxel_grid, method='max'):
    if method == 'max':
        # Max projection along z-axis
        projection = np.max(voxel_grid, axis=0)
    elif method == 'sum':
        projection = np.sum(voxel_grid, axis=0)
        projection = projection / projection.max()
    else:
        raise ValueError("Unknown projection method")
    
    # Convert binary to uint8 for image
    img = (projection * 255).astype(np.uint8)
    return Image.fromarray(img)

for filename in os.listdir(binvox_dir):
    if not filename.endswith(".binvox"):
        continue

    path = os.path.join(binvox_dir, filename)
    with open(path, 'rb') as f:
        model = read_as_3d_array(f)
        voxel = model.data.astype(np.float32)

        image = binvox_to_image(voxel, method='max')  # or 'sum'
        image = image.resize((224, 224))  # Optional: resize for consistency

        # Save image
        out_path = os.path.join(output_dir, filename.replace(".binvox", ".png"))
        image.save(out_path)
        print(f"Saved {out_path}")
