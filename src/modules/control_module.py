import numpy as np
import collections
import mediapipe as mp # Import mediapipe to access PoseLandmark

# Simplified joint limits for a hypothetical robot arm (in radians)
# These would typically come from the robot\'s URDF or specifications.
JOINT_LIMITS = {
    'left_shoulder_x': (-np.pi/2, np.pi/2), # Example range
    'left_shoulder_y': (-np.pi/2, np.pi/2),
    'left_elbow': (0, np.pi), # Elbow can bend from straight (0) to fully bent (pi)
}

class JointTrajectoryGenerator:
    def __init__(self, smoothing_window_size=5, feedback_gain=0.1):
        self.smoothing_window_size = smoothing_window_size
        self.joint_angle_buffers = collections.defaultdict(lambda: collections.deque(maxlen=smoothing_window_size))
        self.feedback_gain = feedback_gain # Proportional gain for feedback

    def _calculate_angle_3d(self, p1, p2, p3):
        """
        Calculates the angle (in radians) between three 3D points (p2 is the vertex).
        Points are expected as (x, y, z) tuples or numpy arrays.
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0 # Avoid division by zero

        angle = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
        return angle

    def map_landmarks_to_joint_commands(self, pose_landmarks):
        """
        Maps MediaPipe pose landmarks to simplified robot joint commands.
        This is a highly simplified mapping for demonstration purposes.

        Args:
            pose_landmarks: A MediaPipe NormalizedLandmarkList object.

        Returns:
            dict: A dictionary of joint names and their target angles (in radians).
        """
        joint_commands = {}

        if not pose_landmarks:
            return joint_commands

        # Extract relevant landmarks for left arm
        # Indices for MediaPipe landmarks (example, refer to MediaPipe documentation for full list)
        # 11: left_shoulder, 13: left_elbow, 15: left_wrist
        # 12: right_shoulder, 14: right_elbow, 16: right_wrist

        # Using left arm for demonstration
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]

        # Convert landmarks to numpy arrays for easier calculation
        ls_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        le_pos = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
        lw_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
        lh_pos = np.array([left_hip.x, left_hip.y, left_hip.z])

        # Calculate left elbow angle
        # Angle at elbow formed by shoulder-elbow vector and wrist-elbow vector
        elbow_angle = self._calculate_angle_3d(ls_pos, le_pos, lw_pos)
        joint_commands['left_elbow'] = np.clip(elbow_angle, *JOINT_LIMITS['left_elbow'])

        # Calculate left shoulder angles (simplified for demonstration)
        # This is a very rough approximation and would need proper kinematic modeling
        # For a more accurate shoulder, you\'d need to consider the orientation of the shoulder joint
        # and potentially use a different set of landmarks or IK.
        # Here, we\'ll try to infer some rotation based on the relative position of elbow to shoulder
        # and shoulder to hip.

        # Vector from hip to shoulder
        hip_to_shoulder = ls_pos - lh_pos
        # Vector from shoulder to elbow
        shoulder_to_elbow = le_pos - ls_pos

        # Simplified shoulder pitch (rotation around x-axis, roughly)
        # Angle between vertical axis (approx. hip-shoulder) and shoulder-elbow vector projected on YZ plane
        # This is a very rough heuristic.
        shoulder_pitch_vec1 = np.array([0, hip_to_shoulder[1], hip_to_shoulder[2]])
        shoulder_pitch_vec2 = np.array([0, shoulder_to_elbow[1], shoulder_to_elbow[2]])
        shoulder_pitch = self._calculate_angle_3d(shoulder_pitch_vec1, np.array([0,0,0]), shoulder_pitch_vec2)
        # Adjust sign based on relative Y position of elbow to shoulder
        if shoulder_to_elbow[1] < 0: # If elbow is \'up\' relative to shoulder in Y (screen coords)
             shoulder_pitch = -shoulder_pitch
        joint_commands['left_shoulder_y'] = np.clip(shoulder_pitch, *JOINT_LIMITS['left_shoulder_y'])

        # Simplified shoulder roll (rotation around y-axis, roughly)
        # Angle between horizontal axis (approx. perpendicular to hip-shoulder) and shoulder-elbow vector projected on XZ plane
        shoulder_roll_vec1 = np.array([hip_to_shoulder[0], 0, hip_to_shoulder[2]])
        shoulder_roll_vec2 = np.array([shoulder_to_elbow[0], 0, shoulder_to_elbow[2]])
        shoulder_roll = self._calculate_angle_3d(shoulder_roll_vec1, np.array([0,0,0]), shoulder_roll_vec2)
        # Adjust sign based on relative X position of elbow to shoulder
        if shoulder_to_elbow[0] > 0: # If elbow is \'out\' relative to shoulder in X (screen coords)
            shoulder_roll = -shoulder_roll
        joint_commands['left_shoulder_x'] = np.clip(shoulder_roll, *JOINT_LIMITS['left_shoulder_x'])

        return joint_commands

    def smooth_joint_commands(self, current_joint_commands):
        """
        Applies a simple moving average filter to joint commands.

        Args:
            current_joint_commands (dict): Dictionary of current joint names and angles.

        Returns:
            dict: Dictionary of smoothed joint names and angles.
        """
        smoothed_commands = {}
        for joint_name, angle in current_joint_commands.items():
            self.joint_angle_buffers[joint_name].append(angle)
            smoothed_commands[joint_name] = np.mean(list(self.joint_angle_buffers[joint_name]))
        return smoothed_commands

    def adjust_commands_with_feedback(self, target_commands, actual_joint_positions):
        """
        Adjusts target commands based on the difference between target and actual joint positions.
        Applies a simple proportional feedback.

        Args:
            target_commands (dict): Dictionary of desired joint names and angles.
            actual_joint_positions (dict): Dictionary of actual joint names and their current angles from PyBullet.

        Returns:
            dict: Adjusted target commands.
        """
        adjusted_commands = target_commands.copy()
        for joint_name, target_angle in target_commands.items():
            if joint_name in actual_joint_positions:
                actual_angle = actual_joint_positions[joint_name]
                error = target_angle - actual_angle
                # Apply proportional feedback: adjust target to reduce error
                adjusted_commands[joint_name] = target_angle + self.feedback_gain * error
                # Re-clip to joint limits after adjustment
                if joint_name in JOINT_LIMITS:
                    adjusted_commands[joint_name] = np.clip(adjusted_commands[joint_name], *JOINT_LIMITS[joint_name])
            # else: # If joint not found in actual_joint_positions, keep original target
            #     print(f"Warning: Actual position for joint {joint_name} not available for feedback.")
        return adjusted_commands

if __name__ == '__main__':
    # This example requires MediaPipe to be running and providing pose_landmarks.
    # For a standalone test, you would need to mock pose_landmarks data.
    print("This module is designed to be integrated with vision_module.py and simulation_module.py.")
    print("Run the main loop from vision_module.py and pass its output here.")
    print("Example usage would look like:")
    print("    from vision_module import capture_and_preprocess_frame, estimate_pose")
    print("    # ... inside a loop ...")
    print("    frame = capture_and_preprocess_frame()")
    print("    landmarks = estimate_pose(frame)")
    print("    if landmarks:")
    print("        generator = JointTrajectoryGenerator()")
    print("        raw_commands = generator.map_landmarks_to_joint_commands(landmarks)")
    print("        smoothed_commands = generator.smooth_joint_commands(raw_commands)")
    print("        print(smoothed_commands)")

    print("Example usage for feedback integration:")
    print("    from vision_module import capture_and_preprocess_frame, estimate_pose")
    print("    from simulation_module import PyBulletSimulation")
    print("    # ... inside a main loop ...")
    print("    sim = PyBulletSimulation(gui=True)")
    print("    sim.load_robot(\"humanoid/humanoid.urdf\") # or other URDF")
    print("    generator = JointTrajectoryGenerator()")
    print("    while True:")
    print("        frame = capture_and_preprocess_frame()")
    print("        landmarks = estimate_pose(frame)")
    print("        if landmarks:")
    print("            raw_commands = generator.map_landmarks_to_joint_commands(landmarks)")
    print("            smoothed_commands = generator.smooth_joint_commands(raw_commands)")

    print("            # Get actual joint positions from PyBullet")
    print("            actual_joint_states = sim.get_joint_states() # This function needs to be added to PyBulletSimulation")
    print("            actual_joint_positions = {name: state[0] for name, state in actual_joint_states.items()}")

    print("            # Adjust commands with feedback")
    print("            final_commands = generator.adjust_commands_with_feedback(smoothed_commands, actual_joint_positions)")

    print("            sim.set_joint_positions(final_commands)")
    print("            sim.step_simulation()")
    print("            # ... add time.sleep if needed ...")


