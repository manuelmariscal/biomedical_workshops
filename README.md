# Biomedical Workshops

## Real-time Gait Analysis for Biomedical Engineers

This repository contains a real-time gait analysis tool using video capture and Mediapipe's pose estimation. The tool is designed to help biomedical engineering students track and analyze gait movements using a laptop camera or a connected mobile phone camera. The analysis is performed in real-time, and various metrics such as angular velocity and angular acceleration are calculated and displayed.

### Features

- Real-time gait analysis using Mediapipe and OpenCV.
- Capture video from a laptop camera or a mobile phone camera.
- Extract and track key points of interest (e.g., ankles, knees, hips, shoulders, elbows, wrists, and center of gravity).
- Calculate and display angular velocities and angular accelerations for each joint.
- Smooth the data using Savitzky-Golay filter.
- Save the data to a CSV file for further analysis.
- Three visualizations:
  - Real-time skeleton movement with tracked points and lines connecting them.
  - Trajectories of key points over time.
  - Fading trails to visualize the pendulum-like movements of each joint.

### Repository Structure

```bash
biomedical_workshops/
├── main.py
├── gait_data.py
├── README.md
├── requirements.txt
└── src/
    └── gait_analysis.py
```

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/manuelmariscal/biomedical_workshops.git
    cd biomedical_workshops
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv bioworkshops
    source bioworkshops/bin/activate  # On Windows, use `bioworkshops\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the real-time gait analysis:

```bash
python main.py --source 0
```

Where `0` is the default source for the laptop camera. You can also use a URL for a mobile phone camera.

#### Arguments

- `--source`: Video source (0 for laptop camera, URL for mobile camera).
- `--test`: Use saved data for testing.

### Data Analysis

The tool captures the video data and extracts key points of interest, saving the results in a CSV file named `gait_data.csv`. It calculates angular velocities and accelerations for each joint and provides three visualizations:

1. **Real-time Skeleton Movement**: Shows the real-time movement of the skeleton with tracked points and lines connecting them.
2. **Trajectories of Key Points**: Displays the trajectories of key points over time.
3. **Fading Trails**: Visualizes the pendulum-like movements of each joint with a fading trail effect.

### Example

To run the analysis with a test dataset:

```bash
python main.py --test
```

### Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Please ensure your code follows the coding standards and includes tests where applicable.

### License

This project is licensed under the MIT License.

### Acknowledgments

This project uses the following libraries:

- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)

---

For more information, please visit the [project repository](https://github.com/manuelmariscal/biomedical_workshops).
