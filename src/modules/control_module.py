import numpy as np
import collections
import mediapipe as mp
import logging



logger = logging.getLogger(__name__)



JOINT_LIMITS = {
    'left_shoulder_x': (-np.pi/2, np.pi/2),
    'left_shoulder_y': (-np.pi/2, np.pi/2),
    'left_elbow': (0, np.pi),
    "left_shoulder_z": (-np.pi/2, np.pi/2),
    "left_forearm_twist": (-np.pi/2, np.pi/2),
    "left_wrist_pitch": (-np.pi/2, np.pi/2),
    "left_wrist_yaw": (-np.pi/2, np.pi/2),
}


NAME_MAP = {
    'left_shoulder_z': 'lbr_iiwa_joint_1',
    'left_shoulder_y': 'lbr_iiwa_joint_2',
    'left_shoulder_x': 'lbr_iiwa_joint_3',
    'left_elbow': 'lbr_iiwa_joint_4',
    'left_forearm_twist': 'lbr_iiwa_joint_5',
    'left_wrist_y': 'lbr_iiwa_joint_6',
    'left_wrist_x': 'lbr_iiwa_joint_7',
}

class JointTrajectoryGenerator:
    def __init__(self, smoothing_window_size=5, feedback_gain=0.1):
        self.smoothing_window_size = smoothing_window_size
        self.joint_angle_buffers = collections.defaultdict(lambda: collections.deque(maxlen=smoothing_window_size))
        self.feedback_gain = feedback_gain

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
            return 0.0

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

        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        logger.info(f"shoulder pose {left_shoulder} | elbow pose {left_elbow} | wrist pose {left_wrist} | hip pose {left_hip} ")
        

        ls_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        le_pos = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
        lw_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
        lh_pos = np.array([left_hip.x, left_hip.y, left_hip.z])

        # Vectors
        v_se = le_pos - ls_pos   # shoulder → elbow
        v_ew = lw_pos - le_pos   # elbow → wrist
        _v_sh = ls_pos - lh_pos   # hip → shoulder

        # Calculate left elbow flexion
        elbow_angle = self._calculate_angle_3d(ls_pos, le_pos, lw_pos)
        joint_commands['left_elbow'] = np.clip(elbow_angle, * JOINT_LIMITS['left_elbow'])

        # Vector from hip to shoulder
        hip_to_shoulder = ls_pos - lh_pos
        # Vector from shoulder to elbow
        shoulder_to_elbow = le_pos - ls_pos

        # Simplified shoulder pitch (rotation around x-axis, roughly)
        # Angle between vertical axis (approx. hip-shoulder) and shoulder-elbow vector projected on YZ plane

        shoulder_pitch_vec1 = np.array([0, hip_to_shoulder[1], hip_to_shoulder[2]])
        shoulder_pitch_vec2 = np.array([0, shoulder_to_elbow[1], shoulder_to_elbow[2]])
        shoulder_pitch = self._calculate_angle_3d(shoulder_pitch_vec1, np.array([0,0,0]), shoulder_pitch_vec2)

        if shoulder_to_elbow[1] < 0: # If elbow is \'up\' relative to shoulder in Y (screen coords)
             shoulder_pitch = -shoulder_pitch
        joint_commands['left_shoulder_y'] = np.clip(shoulder_pitch, *JOINT_LIMITS['left_shoulder_y'])


        # Angle between horizontal axis (approx. perpendicular to hip-shoulder) and shoulder-elbow vector projected on XZ plane
        shoulder_roll_vec1 = np.array([hip_to_shoulder[0], 0, hip_to_shoulder[2]])
        shoulder_roll_vec2 = np.array([shoulder_to_elbow[0], 0, shoulder_to_elbow[2]])
        shoulder_roll = self._calculate_angle_3d(shoulder_roll_vec1, np.array([0,0,0]), shoulder_roll_vec2)
        # Adjust sign based on relative X position of elbow to shoulder
        if shoulder_to_elbow[0] > 0: # If elbow is \'out\' relative to shoulder in X (screen coords)
            shoulder_roll = -shoulder_roll
        joint_commands['left_shoulder_x'] = np.clip(shoulder_roll, *JOINT_LIMITS['left_shoulder_x'])

  
        # 3. Shoulder yaw (rotation around vertical axis)
        
        # estimate yaw as angle between the projection of shoulder→elbow in XZ plane
        v_se_xz = v_se * np.array([1, 0, 1])
        forward_axis = np.array([0, 0, 1])
        yaw = self._calculate_angle_3d(forward_axis, np.zeros(3), v_se_xz)
        if v_se[0] > 0: 
            yaw = -yaw
        
        logger.info(joint_commands)

        joint_commands['left_shoulder_z'] = np.clip(yaw, *JOINT_LIMITS['left_shoulder_z'])

        # 4. Forearm twist (pronation/supination)
        # Compute normal of the plane formed by shoulder-elbow-wrist
        normal = np.cross(v_se, v_ew)
        # Use projection of v_ew onto forearm axis and normal → twist angle
        proj = np.cross(normal, v_se)
        twist = self._calculate_angle_3d(v_ew, np.zeros(3), proj)
        if np.dot(v_ew, normal) < 0: 
            twist = -twist
        joint_commands['left_forearm_twist'] = np.clip(twist, *JOINT_LIMITS['left_forearm_twist'])

        # 5 & 6. Wrist orientation: pitch & yaw (rough)
        hand_dir = v_ew
        pitch = self._calculate_angle_3d(v_ew, np.zeros(3), hand_dir)
        joint_commands['left_wrist_y'] = np.clip(pitch, *JOINT_LIMITS['left_wrist_pitch'])

        horizontal = np.array([hand_dir[0], 0, hand_dir[2]])
        yaw_w = self._calculate_angle_3d(horizontal, np.zeros(3), hand_dir)
        if hand_dir[0] < 0: 
            yaw_w = -yaw_w
        joint_commands['left_wrist_x'] = np.clip(yaw_w, *JOINT_LIMITS['left_wrist_yaw'])

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
                adjusted_commands[joint_name] = target_angle + self.feedback_gain * error
                if joint_name in JOINT_LIMITS:
                    adjusted_commands[joint_name] = np.clip(adjusted_commands[joint_name], *JOINT_LIMITS[joint_name])
            else: 
                logger.warning(f"Warning: Actual position for joint {joint_name} not available for feedback.")
        return adjusted_commands
    
    
    
    def map_and_filter_commands(self,smoothed_commands: dict, actual_joint_positions: dict) -> dict:
        """
        Maps human-readable joint names in `smoothed_commands` to robot joint names
        and filters to only those present in `actual_joint_positions`.

        Args:
            smoothed_commands (dict): Human joint command values.
            actual_joint_positions (dict): Valid robot joint names.

        Returns:
            dict: Filtered and renamed command dictionary.
        """
        filtered = {}
        for cmd_name, value in smoothed_commands.items():
            robot_name = NAME_MAP.get(cmd_name)
            if robot_name and robot_name in actual_joint_positions:
                filtered[robot_name] = value
            else:
                logger.debug(f"No valid robot joint mapped or found for '{cmd_name}'")
        return filtered

if __name__ == '__main__':
    logger.info("This module is designed to be integrated with vision_module.py and simulation_module.py.")
    logger.info("Run the main loop from vision_module.py and pass its output here.")
    logger.info("Example usage would look like:")
    logger.info("    from vision_module import capture_and_preprocess_frame, estimate_pose")
    logger.info("    # ... inside a loop ...")
    logger.info("    frame = capture_and_preprocess_frame()")
    logger.info("    landmarks = estimate_pose(frame)")
    logger.info("    if landmarks:")
    logger.info("        generator = JointTrajectoryGenerator()")
    logger.info("        raw_commands = generator.map_landmarks_to_joint_commands(landmarks)")
    logger.info("        smoothed_commands = generator.smooth_joint_commands(raw_commands)")
    logger.info("        print(smoothed_commands)")

    logger.info("Example usage for feedback integration:")
    logger.info("    from vision_module import capture_and_preprocess_frame, estimate_pose")
    logger.info("    from simulation_module import PyBulletSimulation")
    logger.info("    # ... inside a main loop ...")
    logger.info("    sim = PyBulletSimulation(gui=True)")
    logger.info("    sim.load_robot(\"humanoid/humanoid.urdf\") # or other URDF")
    logger.info("    generator = JointTrajectoryGenerator()")
    logger.info("    while True:")
    logger.info("        frame = capture_and_preprocess_frame()")
    logger.info("        landmarks = estimate_pose(frame)")
    logger.info("        if landmarks:")
    logger.info("            raw_commands = generator.map_landmarks_to_joint_commands(landmarks)")
    logger.info("            smoothed_commands = generator.smooth_joint_commands(raw_commands)")

    logger.info("            # Get actual joint positions from PyBullet")
    logger.info("            actual_joint_states = sim.get_joint_states() # This function needs to be added to PyBulletSimulation")
    logger.info("            actual_joint_positions = {name: state[0] for name, state in actual_joint_states.items()}")

    logger.info("            # Adjust commands with feedback")
    logger.info("            final_commands = generator.adjust_commands_with_feedback(smoothed_commands, actual_joint_positions)")

    logger.info("            sim.set_joint_positions(final_commands)")
    logger.info("            sim.step_simulation()")
    logger.info("            # ... add time.sleep if needed ...")


