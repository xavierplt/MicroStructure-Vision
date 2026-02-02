# MicroStructure Vision

**Automated Metallographic Grain Analysis using Classical Computer Vision**

This project provides a robust tool for analyzing microstructure images of steel samples. It utilizes classical computer vision algorithms to segment grains, calculate ASTM E112 G-numbers, and estimate carbon content.

## Features

- **Dual Algorithms**: compare results between **Otsu's Thresholding** and **Marker-based Watershed** segmentation.
- **Grain Metrics**:
    - **Grain Count**: Total number of detected grains.
    - **ASTM G-Number**: Standard grain size number ($G = 3.322 \log_{10}(N_a) - 2.95$).
    - **Grain Size Distribution**: Visual histogram of grain areas.
- **Carbon Content Estimation**: Approximate carbon percentage based on dark phase analysis.
- **Batch Processing**: Process entire datasets efficiently.
- **Visualizations**: Overlay segmentation boundaries and generate comparative plots.

## Project Structure

- `microstructure_analysis_app.py`: Main interactive Streamlit application.
- `src/`: Core logic modules.
    - `segmentation_algorithms.py`: Otsu and Watershed implementations.
    - `grain_metrics_calculation.py`: Analysis functions (G-number, Carbon).
    - `image_handling.py`: Utilities for loading and processing images.
- `batch_process.py`: Script to process all images in `Data/` and save results to `Results/`.
- `visualize_results.py`: Script to generate global statistical plots from batch results.
- `Data/`: Input images folder.
- `Results/`: Output folder for segmentation masks, comparison images, and CSV reports.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/xavierplt/Projet.git
    cd Projet
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Interactive Web App

Run the Streamlit app to analyze images one by one:

```bash
streamlit run microstructure_analysis_app.py
```

- Upload an image or select from the `Data` folder.
- Tune parameters (Kernel Size, Min Distance, Invert Mask).
- View instant results and visualizations.

### Batch Processing

To process the entire dataset (e.g., all images in `Data/`):

1.  Run the processing script:
    ```bash
    python batch_process.py
    ```
    This generates `_comparison.png` for every image and a `batch_results.csv` summary.

2.  Generate global analysis plots:
    ```bash
    python visualize_results.py
    ```
    Outputs plots to `Results/Plots`.

## Results

Sample of generated analysis:
- **Comparison Scatter Plot**: Otsu vs Watershed G-number consistency.
- **Grain Size Histograms**: Distribution of grain sizes across the dataset.
- **Segmentation Overlays**: Visual quality check of grain boundaries.

## Author

Project developed for Visual Computing - Master IA.
