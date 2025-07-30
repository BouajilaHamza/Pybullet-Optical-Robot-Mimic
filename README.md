# PyBullet Humanoid Teleoperation via Computer Vision

A real-time teleoperation system that enables controlling a PyBullet humanoid robot using computer vision and pose estimation.

## 🚀 Features

- **Real-time Pose Estimation**: Uses MediaPipe for accurate human pose detection
- **Modular Architecture**: Separated into vision, inference, control, and simulation modules
- **Smooth Motion Control**: Implements joint trajectory generation with smoothing
- **Physics Simulation**: Leverages PyBullet for realistic physics and environment interaction
- **Feedback Loop**: Maintains stability and accuracy through continuous error correction

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Pybullet-Optical-Robot-Mimic.git
   cd Pybullet-Optical-Robot-Mimic
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Project Structure

```
src/
├── modules/
│   ├── vision_module.py     # Camera capture and pose estimation
│   ├── control_module.py    # Joint trajectory generation and control
│   └── simulation_module.py # PyBullet simulation environment
├── models/                  # Humanoid URDF models
└── utils/                   # Utility functions and helpers
```

## 🚀 Quick Start

1. Ensure your webcam is connected and accessible
2. Run the main teleoperation script:
   ```bash
   python src/main.py
   ```
3. Position yourself in front of the camera
4. The humanoid in the simulation should now mimic your movements

## 🎯 Key Components

### Vision Module
- Captures and processes camera frames
- Performs pose estimation using MediaPipe
- Extracts keypoints for human pose

### Control Module
- Maps human poses to robot joint commands
- Implements smoothing for natural movements
- Includes balance and collision avoidance

### Simulation Module
- Manages the PyBullet physics simulation
- Handles robot model loading and control
- Provides visualization and debugging tools

## 📊 Performance Metrics

- **Latency**: Measures the end-to-end delay in the control loop
- **Accuracy**: Tracks joint angle errors between target and actual poses
- **Stability**: Monitors balance and collision occurrences

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [PyBullet Documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [URDF Documentation](http://wiki.ros.org/urdf)