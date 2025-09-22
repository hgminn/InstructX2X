import os
import pandas as pd
import numpy as np
from PIL import Image

def load_and_filter_data(metadata_path, bb_path):
    """
    Loads MIMIC-CXR data and filters for PA (Posteroanterior) view images only.
    """
    mimic_meta = pd.read_csv(metadata_path)
    mimic_bb = pd.read_csv(bb_path)
    
    # Select PA view images
    mimic_meta_pa = mimic_meta[mimic_meta['ViewPosition'] == 'PA']
    pa_dicom_ids = mimic_meta_pa['dicom_id'].unique()
    
    # Filter the bounding box data based on the selected dicom_ids
    filtered_mimic_bb = mimic_bb[mimic_bb['dicom_id'].isin(pa_dicom_ids)].reset_index(drop=True)
    
    return filtered_mimic_bb

def resize_bounding_boxes(df, target_size=512):
    """
    Resizes bounding box coordinates to fit the specified target image size.
    """
    resized_df = df.copy()
    
    resized_df['x'] = (df['x'] * target_size / df['image_width']).round().astype(int)
    resized_df['y'] = (df['y'] * target_size / df['image_height']).round().astype(int)
    resized_df['w'] = (df['w'] * target_size / df['image_width']).round().astype(int)
    resized_df['h'] = (df['h'] * target_size / df['image_height']).round().astype(int)
    
    resized_df['image_width'] = target_size
    resized_df['image_height'] = target_size
    
    return resized_df

def create_binary_mask_from_all_overlaps(boxes_df, category, target_size=512):
    """
    Creates a binary mask by overlapping all bounding boxes for a given category.
    Any pixel covered by at least one box is included in the mask.
    """
    category_boxes = boxes_df[boxes_df['category_name'] == category]
    overlap_count = np.zeros((target_size, target_size), dtype=np.uint16)

    for _, box in category_boxes.iterrows():
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        # Clip coordinates to stay within image boundaries
        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(target_size, x + w), min(target_size, y + h)
        
        if x_start < x_end and y_start < y_end:
            overlap_count[y_start:y_end, x_start:x_end] += 1
    
    # Set all pixels covered by at least one bounding box to 1
    final_mask = (overlap_count >= 1).astype(np.uint8)
    
    return final_mask

if __name__ == "__main__":
    # --- Configuration ---
    METADATA_PATH = 'mimic-cxr-2.0.0-metadata.csv'
    BB_PATH = 'MS_CXR_Local_Alignment_v1.1.0.csv'
    OUTPUT_DIR = 'pseudo_masks_binary_all_overlaps'
    TARGET_SIZE = 512
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Processing and Mask Generation ---
    print("1. Loading and filtering PA view data...")
    filtered_bb_df = load_and_filter_data(METADATA_PATH, BB_PATH)

    print(f"2. Resizing bounding boxes to {TARGET_SIZE}x{TARGET_SIZE}...")
    resized_bb_df = resize_bounding_boxes(filtered_bb_df, target_size=TARGET_SIZE)

    categories = resized_bb_df['category_name'].unique()
    print(f"3. Found {len(categories)} categories. Generating masks...")

    for category in categories:
        print(f"  - Processing: {category}")
        
        # Create the binary mask
        binary_mask = create_binary_mask_from_all_overlaps(resized_bb_df, category, target_size=TARGET_SIZE)
        
        # Sanitize category name for file path
        safe_category_name = category.replace(" ", "_").replace("/", "_")
        
        # Save the mask as a .npy file
        npy_path = os.path.join(OUTPUT_DIR, f'mask_{safe_category_name}.npy')
        np.save(npy_path, binary_mask)
        
        # Save the mask as a .png image for visualization
        # 0 (background) -> black, 1 (mask) -> white
        mask_img = Image.fromarray(binary_mask * 255, mode='L')
        png_path = os.path.join(OUTPUT_DIR, f'mask_{safe_category_name}.png')
        mask_img.save(png_path)

    print(f"\nMask generation complete. Results saved in: {OUTPUT_DIR}")