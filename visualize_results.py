import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_results(csv_path="Results/batch_results.csv", output_dir="Results/Plots"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run batch_process.py first.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Comparison of G-Numbers (Otsu vs Watershed)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Otsu_G'], df['Watershed_G'], alpha=0.7, color='purple', edgecolors='k')
    
    # Plot y=x line for reference
    min_val = min(df['Otsu_G'].min(), df['Watershed_G'].min())
    max_val = max(df['Otsu_G'].max(), df['Watershed_G'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Perfect Agreement)')
    
    plt.title('Comparison of ASTM G-Number: Otsu vs Watershed')
    plt.xlabel('Otsu G-Number')
    plt.ylabel('Watershed G-Number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G_Number_Comparison.png'))
    print(f"Saved G_Number_Comparison.png")
    
    # 2. Grain Count Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    bins = np.linspace(min(df['Otsu_Count'].min(), df['Watershed_Count'].min()), 
                       max(df['Otsu_Count'].max(), df['Watershed_Count'].max()), 30)
    
    plt.hist(df['Otsu_Count'], bins=bins, alpha=0.6, label='Otsu', color='blue', edgecolor='black')
    plt.hist(df['Watershed_Count'], bins=bins, alpha=0.6, label='Watershed', color='green', edgecolor='black')
    
    plt.title('Distribution of Grain Counts')
    plt.xlabel('Number of Grains Detected')
    plt.ylabel('Frequency (Number of Images)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Grain_Count_Distribution.png'))
    print(f"Saved Grain_Count_Distribution.png")

    # 3. Carbon Content by Steel Type (if folder name implies type)
    # Extract "Family" from File path (e.g., "Carbone Steel\image.jpg" -> "Carbone Steel")
    df['Family'] = df['File'].apply(lambda x: os.path.dirname(x).split(os.sep)[0] if os.sep in x else 'Unknown')
    
    families = df['Family'].unique()
    
    plt.figure(figsize=(12, 6))
    
    # Boxplot for Carbon Content (Otsu estimates)
    # Prepare data for boxplot
    data_to_plot = [df[df['Family'] == fam]['Otsu_Carbon'] for fam in families]
    
    plt.boxplot(data_to_plot, labels=families)
    plt.title('Estimated Carbon Content (%) by Steel Family (Otsu)')
    plt.ylabel('Carbon Content (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Carbon_Content_Boxplot.png'))
    print(f"Saved Carbon_Content_Boxplot.png")
    
    print("Visualization complete.")

if __name__ == "__main__":
    visualize_results()
