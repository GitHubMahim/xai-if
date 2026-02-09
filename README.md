# Influence Visualization Tools

Seven interactive tools for analyzing influences in continuous field reconstruction models:

1. **Region Influence Tool** (`plot_region_influences.py`) - Region-to-region influences
2. **Point Influence Tool** (`plot_point_influences.py`) - Point-to-point influences  
3. **Temporal Influence Tool** (`plot_influence_over_time.py`) - How influences change over time
4. **Time Line Graph Tool** (`plot_time_line_graph.py`) - Clustered time series of influences
5. **Region Matrix Tool** (`plot_region_matrix.py`) - 54x54 region influence matrix
6. **Region Bar Graph Tool** (`plot_region_bar_graph.py`) - Bidirectional region influence bars
7. **Region Histogram Tool** (`plot_region_histogram.py`) - Distance distribution histogram

## Installation
1. Clone the project and cd into the folder.

```bash
git clone https://github.com/celiafish/xai-if.git
cd xai-if
```

2. Install the required libraries to your environment.

```bash
pip install -r requirements.txt
```

3. Download and unzip the data file. (Only needed for Point Influence Tool)

Make sure the data file is in the root directory.
| Dataset | Link |
| -------------------------------------------- | ------------------------------------------------------------ |
| All test data influences | [[Google Drive]](https://drive.google.com/file/d/1mrEu7sJ3Dc1AsR8pLDCdLtgLO973518f/view?usp=sharing) |


---

## 1. Region Influence Tool

**Files needed:** `regions_data.npy`, `regions_data_positive.npy`, `regions_data_negative.npy`, `yrefs.npy`

**What it does:** Shows how different spatial regions influence each other across 54 regions.

**4 Maps:**
- **Top Left:** Click to select a region
- **Top Right:** Shows which regions influence the selected region (purple)
- **Bottom Left:** Total influence for all regions (blue, static)
- **Bottom Right:** Shows which regions the selected region influences (grey)

**Controls:** Click regions, Overall/Positive/Negative buttons, percentage slider (10-100%), +/- buttons

**Run:**
```bash
python plot-scripts/plot_region_influences.py
```

---

## 2. Point Influence Tool

**Files needed:** `test_coords.npy`, `train_coords.npy`, `yrefs.npy`, `influences_all_test.npy`

**What it does:** Shows which individual training points influence a specific test location.

**2 Maps:**
- **Left:** Click to select test location (yellow circle)
- **Right:** Shows influential training points (green dots)

**Controls:** Click locations, coordinate sliders (longitude/latitude), All/Positive/Negative buttons, percentage slider (10-100%), +/- buttons

**Run:**
```bash
python plot-scripts/plot_point_influences.py
```

---

## 3. Temporal Influence Tool

**Files needed:** `test_coords.npy`, `train_coords.npy`, `yrefs.npy`, `index_15533_influences.npy`, `index_15533_global_influence.npy`

**What it does:** Shows how influences change over time for a fixed test point (index 15533).

**2 Maps:**
- **Left:** Reference field with fixed test point (yellow circle)
- **Right:** Influential training points at current timestamp (green dots)

**Controls:** Timestamp slider, All/Positive/Negative/Global buttons, percentage slider (10-100%), +/- buttons, dataset-wide influence metrics

**Run:**
```bash
python plot-scripts/plot_influence_over_time.py
```

---

## 4. Time Line Graph Tool

**Files needed:** `index_15533_influences.npy`

**What it does:** Shows clustered training point influences over time as a line graph. Uses PCA + K-means clustering to reduce visual clutter from 2765 training points to 20 representative clusters.

**1 Graph:**
- **Line Graph:** Each line represents a cluster of training points, showing how their influence changes over time

**Controls:** Drag to zoom on X-axis, Reset button to restore full view

**Run:**
```bash
python plot-scripts/plot_time_line_graph.py
```

---

## 5. Region Matrix Tool

**Files needed:** `regions_data.npy`, `regions_data_positive.npy`, `regions_data_negative.npy`

**What it does:** Creates a 54x54 heatmap matrix showing all pairwise region-to-region influences. Cell (i, j) shows the influence of region i on region j.

**1 Matrix:**
- **Heatmap:** 54x54 matrix with color-coded influence values, grid lines for readability

**Controls:** Overall/Positive/Negative buttons to filter influence types

**Run:**
```bash
python plot-scripts/plot_region_matrix.py
```

---

## 6. Region Bar Graph Tool

**Files needed:** `regions_data.npy`, `regions_data_positive.npy`, `regions_data_negative.npy`

**What it does:** Shows bidirectional region influences as a bar graph. Top bars show influence OF selected region on others, bottom bars show influence ON selected region from others.

**1 Graph:**
- **Bar Graph:** Dual-directional bars (top: influence OF, bottom: influence ON) for all 54 regions

**Controls:** Region slider (0-53), Overall/Positive/Negative buttons, +/- buttons for region selection

**Run:**
```bash
python plot-scripts/plot_region_bar_graph.py
```

---

## 7. Region Histogram Tool

**Files needed:** `regions_data.npy`

**What it does:** Shows the distribution of distances (in region hops) from a selected region to other regions. Distance 0 = self, 1 = neighbors, 2 = two hops away, etc.

**1 Histogram:**
- **Histogram:** X-axis shows distance in regions, Y-axis shows number of regions at that distance

**Controls:** Region slider (0-53), Top 10%/50%/100% buttons to filter regions, +/- buttons for region selection

**Run:**
```bash
python plot-scripts/plot_region_histogram.py
```

---

## Troubleshooting

- **FileNotFoundError:** Check all required .npy files are present in the `tool-data` folder
- **Empty plots:** Verify influence data contains valid values
- **ImportError:** Run `pip install -r requirements.txt`
- **Coordinates:** Must be normalized between 0 and 1

## Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Ensure data files are in the `tool-data` folder
3. Run any tool: `python plot-scripts/plot_[tool_name].py`
4. Use interactive controls to explore influences

---

## Computing Influences Yourself

To compute influences from scratch using the MMGN model, download the complete folder with all necessary files and models:

| Resource | Link |
| -------------------------------------------- | ------------------------------------------------------------ |
| MMGN Model & Influence Computation Files | [[Google Drive]](https://drive.google.com/file/d/1q_CcZQ6GokolvIDcNKZGHPGB1EV1hhCJ/view?usp=sharing) |

**Setup Instructions:**

1. **Build the MMGN environment:**
   ```bash
   conda env create -f mmgn_env.yml
   conda activate mmgn
   ```

2. **Run the influence computation:**
   ```bash
   python test_influence.py
   ```

**Important Notes:**
- The `test_influence.py` file is the main entry point for computing influences. It calls other methods and models from the other files in the folder.
- The `mmgn_env.yml` file contains the conda environment configuration needed to ensure all dependencies are properly installed.
- Make sure to activate the MMGN environment before running `test_influence.py`.


***For any questions or issues, please contact mohtadimahim@gmail.com***
