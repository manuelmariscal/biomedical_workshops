# Biomedical Workshops

## Real-time Movement Analysis for Biomedical Engineers

This repository contains a real-time movement analysis tool using video capture and Mediapipe's pose estimation. The tool is designed to help biomedical engineering students track and analyze movements using a laptop camera or a connected mobile phone camera. The analysis can be performed in real-time or on recorded data, and various metrics and visualizations are provided.

### Features

- **Real-time movement tracking** using Mediapipe and OpenCV.
- **Capture video** from a laptop camera or a mobile phone camera.
- **Extract and track key points** of interest (e.g., ankles, knees, hips, shoulders, elbows, wrists, and center of gravity).
- **Save tracked data** to CSV files for further analysis.
- **Analyze movement data** with various analytical tools.
- **Calculate correlations** between movements.
- **Visualize correlation matrices** and skeleton heatmaps.
- **Multiple modes of operation**:
  - **track**: Capture and track movements in real-time.
  - **analyze**: Analyze saved movement data.
  - **corr**: Compute and visualize correlations between movement datasets.

### Repository Structure

```bash
biomedical_workshops/
├── main.py
├── README.md
├── requirements.txt
└── src/
    ├── tracking.py
    ├── corr.py
    └── analysis.py
```

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/manuelmariscal/biomedical_workshops.git
    cd biomedical_workshops
    ```

2. **Create a virtual environment** and activate it:

    ```bash
    python -m venv bioworkshops
    # On Windows:
    bioworkshops\Scripts\activate
    # On Unix or MacOS:
    source bioworkshops/bin/activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

The tool operates in different modes: `track`, `analyze`, and `corr`. You can specify the mode using the `--mode` argument.

#### Tracking Mode (`track`)

To run the real-time movement tracking:

```bash
python main.py --mode track --source 0
```

Where `0` is the default source for the laptop camera. You can also use a URL or another device index for a mobile phone camera or an external camera.

#### Analysis Mode (`analyze`)

To analyze saved movement data:

```bash
python main.py --mode analyze --data_folder data
```

This mode processes the CSV files in the specified `data_folder` and performs analysis, including plotting movements and calculating metrics.

#### Correlation Mode (`corr`)

To compute and visualize correlations between movement datasets:

```bash
python main.py --mode corr --data_folder data
```

This mode processes the CSV files in the specified `data_folder`, computes correlation matrices between pairs of datasets, and visualizes the results, including skeleton heatmaps.

#### Arguments

- `--mode`: Mode of operation (`track`, `analyze`, or `corr`).
- `--source`: Video source (e.g., `0` for laptop camera, URL for mobile camera). Used only in `track` mode.
- `--data_folder`: Folder where CSV data files are stored. Used in `analyze` and `corr` modes.

### Data Analysis

The tool captures video data and extracts key points of interest, saving the results in CSV files within the specified `data_folder`. It provides various visualizations and analyses, depending on the mode:

1. **Tracking Mode (`track`)**:
   - **Real-time skeleton movement**: Displays the real-time movement of the skeleton with tracked key points and lines connecting them.
   - **Data Saving**: Saves the tracked data to CSV files for later analysis.

2. **Analysis Mode (`analyze`)**:
   - **Movement Plots**: Visualizes the movements from the saved data.
   - **Animations**: Creates animations comparing valid and invalid movements.
   - **Correlation Calculations**: Computes correlations between movements.

3. **Correlation Mode (`corr`)**:
   - **Correlation Matrices**: Generates and displays correlation matrices for each pair of movement datasets.
   - **Skeleton Heatmaps**: Plots the skeleton with a heatmap based on average keypoint correlations.
     - **Correct Orientation**: The skeleton is displayed with the correct orientation.
     - **Enhanced Visualization**: Thicker lines and larger keypoints are used for better visibility.
     - **Blue Color Map**: A blue color map is used for the heatmap.
   - **Visualization of Results**: Both the correlation matrices and the skeleton heatmap are displayed simultaneously with appropriate titles.

### Example

To run the tool in `corr` mode and analyze data in the `data` folder:

```bash
python main.py --mode corr --data_folder data
```

### Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Please ensure your code follows the coding standards and includes tests where applicable.

### License

This project is licensed under the MIT License.

### Acknowledgments

This project uses the following libraries:

- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)

---

For more information, please visit the [project repository](https://github.com/manuelmariscal/biomedical_workshops).