import os
import numpy as np
from PIL import Image
from tqdm import tqdm

TARGET_SIZE = (128, 128)

def normalize_and_resize_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0  
    return img_np


    parts = shadowed_name.split('_')
    return '_'.join(parts[-3:])  #

def compute_l2_distance(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))  # Value ∈ [0, 1]

def compute_l2_distances(clean_dir, shadow_dir):
    shadow_files = [f for f in os.listdir(shadow_dir) if f.endswith(".png")]
    
    l2_distances = []
    missing_count = 0
    
    for shadowed_name in tqdm(shadow_files, desc="Computing L2 Distances"):
        clean_name = extract_clean_filename(shadowed_name)
        shadow_path = os.path.join(shadow_dir, shadowed_name)
        clean_path = os.path.join(clean_dir, clean_name)

        if not os.path.exists(clean_path):
            missing_count += 1
            continue

        shadow_img = normalize_and_resize_img(shadow_path)
        clean_img = normalize_and_resize_img(clean_path)

        l2 = compute_l2_distance(shadow_img, clean_img)
        l2_distances.append((shadowed_name, clean_name, l2))

    if not l2_distances:
        print("No valid image pairs found.")
    else:
        avg_l2 = np.mean([x[2] for x in l2_distances])
        print(f"\n✅ Average L2 distance (normalized): {avg_l2:.4f}")
        print(f"📸 Total matched image pairs: {len(l2_distances)}")
        print(f"❌ Missing clean images: {missing_count}")



if __name__ == "__main__":
    compute_l2_distances(clean_dir, shadow_dir)
