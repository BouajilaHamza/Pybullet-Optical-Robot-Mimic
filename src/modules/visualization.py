import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

class JointAngleVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.time_data = deque(maxlen=max_points)
        self.target_angles = deque(maxlen=max_points)
        self.actual_angles = deque(maxlen=max_points)
        
        # Set up the plot
        plt.style.use('ggplot')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line_target, = self.ax.plot([], [], 'r-', label='Target Angle')
        self.line_actual, = self.ax.plot([], [], 'b-', label='Actual Angle')
        
        # Customize the plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Angle (rad)')
        self.ax.set_title('Elbow Joint Angle Tracking')
        self.ax.legend()
        self.ax.grid(True)
        
        self.start_time = time.time()
        self.running = True
        
        # Start the animation in a separate thread
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
    
    def add_data(self, target_angle, actual_angle):
        """Add new data points to the plot."""
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.target_angles.append(target_angle)
        self.actual_angles.append(actual_angle)
    
    def update_plot(self, frame):
        """Update the plot with new data."""
        if len(self.time_data) > 0:
            self.line_target.set_data(self.time_data, self.target_angles)
            self.line_actual.set_data(self.time_data, self.actual_angles)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def close(self):
        """Close the plot window."""
        self.running = False
        plt.close(self.fig)

# Global instance for easy access
visualizer = None

def init_visualizer():
    """Initialize the visualizer."""
    global visualizer
    if visualizer is None:
        visualizer = JointAngleVisualizer()

def update_visualizer(target_angle, actual_angle):
    """Update the visualizer with new data."""
    global visualizer
    if visualizer is not None:
        visualizer.add_data(target_angle, actual_angle)

def close_visualizer():
    """Close the visualizer."""
    global visualizer
    if visualizer is not None:
        visualizer.close()
        visualizer = None
