import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.image_handling import load_image_from_path, overlay_mask
from src.segmentation_algorithms import apply_otsu_segmentation, apply_watershed_segmentation
from src.grain_metrics_calculation import calculate_grain_metrics, calculate_g_number, calculate_carbon_content

def batch_process(data_dir="Data", output_dir="Results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results_data = []
    
    # Walk through data directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, data_dir)
                
                print(f"Processing {rel_path}...")
                
                # Load image
                image_rgb, image_gray = load_image_from_path(file_path)
                if image_gray is None:
                    continue
                    
                # Standard parameters (can be tuned or we could run multiple variations)
                # Using defaults or what seemed to work generally
                # Note: We can add 'invert' logic if we detect it, but for batch, maybe try both or stick to standard?
                # Let's assume standard for now, or check mean intensity?
                # Heuristic: if mean < 128, maybe dark background? But microstructure is complex.
                # Let's use standard parameters.
                
                # Otsu
                otsu_labels, _, _ = apply_otsu_segmentation(image_gray, kernel_size=3)
                otsu_count, otsu_areas = calculate_grain_metrics(otsu_labels)
                otsu_g = calculate_g_number(otsu_count, 1.0) # Assuming 1mm2 for G-number comparison if scale unknown
                otsu_c, _ = calculate_carbon_content(image_gray, otsu_labels)
                
                # Watershed
                ws_labels, _ = apply_watershed_segmentation(image_gray, min_distance=20)
                ws_count, ws_areas = calculate_grain_metrics(ws_labels)
                ws_g = calculate_g_number(ws_count, 1.0)
                ws_c, _ = calculate_carbon_content(image_gray, ws_labels)
                
                # Visualization
                from skimage.segmentation import find_boundaries
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                axes[0].imshow(image_rgb)
                axes[0].set_title(f"Original: {file}")
                axes[0].axis('off')
                
                # Otsu
                otsu_bound = find_boundaries(otsu_labels, mode='thick')
                otsu_viz = overlay_mask(image_rgb, otsu_bound, color=(255, 0, 0), alpha=1.0)
                axes[1].imshow(otsu_viz)
                axes[1].set_title(f"Otsu (N={otsu_count}, G={otsu_g:.2f})")
                axes[1].axis('off')
                
                # Watershed
                ws_bound = find_boundaries(ws_labels, mode='thick')
                ws_viz = overlay_mask(image_rgb, ws_bound, color=(0, 255, 0), alpha=1.0)
                axes[2].imshow(ws_viz)
                axes[2].set_title(f"Watershed (N={ws_count}, G={ws_g:.2f})")
                axes[2].axis('off')
                
                # Save comparison image
                # Create subdirs in results if needed to mirror structure
                sub_dir = os.path.dirname(rel_path)
                save_dir = os.path.join(output_dir, sub_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                save_name = os.path.splitext(file)[0] + "_comparison.png"
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, save_name))
                plt.close(fig)
                
                # Append data
                results_data.append({
                    "File": rel_path,
                    "Otsu_Count": otsu_count,
                    "Otsu_G": otsu_g,
                    "Otsu_Carbon": otsu_c,
                    "Otsu_MeanArea": np.mean(otsu_areas) if len(otsu_areas)>0 else 0,
                    "Watershed_Count": ws_count,
                    "Watershed_G": ws_g,
                    "Watershed_Carbon": ws_c,
                    "Watershed_MeanArea": np.mean(ws_areas) if len(ws_areas)>0 else 0
                })

    # Save CSV
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "batch_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Batch processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    batch_process()
