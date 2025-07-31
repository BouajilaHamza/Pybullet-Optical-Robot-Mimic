import numpy as np
import collections
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

# Define joint limits for human-readable joint names.
# These limits should correspond to the expected range of motion for the KUKA iiwa joints.
# Adjust these values based on your robot's specific configuration and safe operating ranges.
JOINT_LIMITS = {
    'left_shoulder_x': (-np.pi/2, np.pi/2), # Corresponds to KUKA J3 (Shoulder Roll)
    'left_shoulder_y': (-np.pi/2, np.pi/2), # Corresponds to KUKA J2 (Shoulder Pitch)
    'left_elbow': (0, np.pi),               # Corresponds to KUKA J4 (Elbow Flexion)
    "left_shoulder_z": (-np.pi/2, np.pi/2), # Corresponds to KUKA J1 (Shoulder Yaw)
    "left_forearm_twist": (-np.pi/2, np.pi/2), # Corresponds to KUKA J5 (Forearm Twist)
    "left_wrist_pitch": (-np.pi/2, np.pi/2),   # Corresponds to KUKA J6 (Wrist Pitch)
    "left_wrist_yaw": (-np.pi/2, np.pi/2),     # Corresponds to KUKA J7 (Wrist Yaw)
}

# Map human-readable joint names to KUKA iiwa PyBullet joint names.
NAME_MAP = {
    'left_shoulder_z': 'lbr_iiwa_joint_1',
    'left_shoulder_y': 'lbr_iiwa_joint_2',
    'left_shoulder_x': 'lbr_iiwa_joint_3',
    'left_elbow': 'lbr_iiwa_joint_4',
    'left_forearm_twist': 'lbr_iiwa_joint_5',
    'left_wrist_y': 'lbr_iiwa_joint_6', # Mapping left_wrist_y to KUKA J6 (Pitch)
    'left_wrist_x': 'lbr_iiwa_joint_7', # Mapping left_wrist_x to KUKA J7 (Yaw)
}

class JointTrajectoryGenerator:
    """
    Generates robot joint commands from MediaPipe pose landmarks,
    including smoothing and feedback adjustment.
    """
    def __init__(self, smoothing_window_size=5, feedback_gain=0.1):
        """
        Initializes the JointTrajectoryGenerator.

        Args:
            smoothing_window_size (int): Size of the moving average window for smoothing.
            feedback_gain (float): Proportional gain for feedback control.
        """
        self.smoothing_window_size = smoothing_window_size
        # Buffers to store historical joint angles for smoothing
        self.joint_angle_buffers = collections.defaultdict(lambda: collections.deque(maxlen=smoothing_window_size))
        self.feedback_gain = feedback_gain

    def _calculate_angle_3d(self, p1, p2, p3):
        """
        Calculates the angle (in radians) between three 3D points (p2 is the vertex).
        Points are expected as (x, y, z) tuples or numpy arrays.
        Returns 0.0 if any vector has zero magnitude to prevent division by zero.
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        # Clip the argument to arccos to prevent numerical errors outside [-1, 1]
        angle = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
        return angle

    def map_landmarks_to_joint_commands(self, pose_landmarks):
        """
        Maps MediaPipe pose landmarks to robot joint commands.

        Args:
            pose_landmarks: MediaPipe pose landmarks object.

        Returns:
            dict: Dictionary of computed joint names and angles.
        """
        joint_commands = {}
        if not pose_landmarks:
            logger.warning("No pose landmarks detected. Returning empty commands.")
            return joint_commands

        # Extract relevant landmark positions
        ls = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        le = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        lw = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

        p_sh = np.array([ls.x, ls.y, ls.z])
        p_el = np.array([le.x, le.y, le.z])
        p_wr = np.array([lw.x, lw.y, lw.z])

        # --- Step 1: Normalize all points relative to the shoulder ---
        # This makes the human's shoulder the origin (0,0,0) for subsequent calculations,
        # mimicking the KUKA iiwa's fixed base.
        p_el_norm = p_el - p_sh # Elbow position relative to shoulder
        p_wr_norm = p_wr - p_sh # Wrist position relative to shoulder
        
        # Define vectors based on these normalized points
        # Vector from shoulder (origin) to elbow
        v_upper_arm = p_el_norm
        # Vector from elbow to wrist
        v_forearm = p_wr_norm - p_el_norm 

        # Ensure segment vectors are not zero length before normalizing to unit vectors
        if np.linalg.norm(v_upper_arm) == 0 or np.linalg.norm(v_forearm) == 0:
            logger.warning("One or more arm segments have zero length. Skipping some joint calculations.")
            return joint_commands 

        n_upper_arm = v_upper_arm / np.linalg.norm(v_upper_arm) # Unit vector: Shoulder -> Elbow
        n_forearm = v_forearm / np.linalg.norm(v_forearm)     # Unit vector: Elbow -> Wrist

        # --- Step 2: Calculate KUKA iiwa Joint Angles ---

        # 1. KUKA Joint 4 (Elbow Flexion)
        # This is the angle at the elbow, formed by the upper arm and forearm segments.
        # Vertex: p_el_norm (elbow)
        # Points: np.zeros(3) (shoulder, which is now the origin) and p_wr_norm (wrist)
        elbow_angle = self._calculate_angle_3d(np.zeros(3), p_el_norm, p_wr_norm)
        joint_commands['left_elbow'] = np.clip(elbow_angle, *JOINT_LIMITS['left_elbow'])

        # Coordinate system transformation from MediaPipe to KUKA base frame
        # MediaPipe's default camera frame: X-right, Y-down, Z-into screen
        # A common KUKA iiwa base frame: X-forward, Y-left, Z-up
        # Transformation mapping:
        # KUKA_X = -MediaPipe_Z
        # KUKA_Y = +MediaPipe_X
        # KUKA_Z = -MediaPipe_Y
        
        n_upper_arm_kuka = np.array([-n_upper_arm[2], n_upper_arm[0], -n_upper_arm[1]])
        n_forearm_kuka = np.array([-n_forearm[2], n_forearm[0], -n_forearm[1]])

        # 2. KUKA Joint 1 (Shoulder Yaw) - `lbr_iiwa_joint_1`
        # Rotation around the Z-axis of the KUKA base.
        # This is the angle of the upper arm vector's projection on the XY plane (of KUKA base)
        # with respect to KUKA's X-axis (forward).
        shoulder_yaw = np.arctan2(n_upper_arm_kuka[1], n_upper_arm_kuka[0])
        joint_commands['left_shoulder_z'] = np.clip(shoulder_yaw, *JOINT_LIMITS['left_shoulder_z'])

        # 3. KUKA Joint 2 (Shoulder Pitch) - `lbr_iiwa_joint_2`
        # Rotation around the Y-axis of the KUKA base (after J1 rotation).
        # This is the angle of the upper arm vector with the horizontal (XY) plane.
        shoulder_pitch = np.arctan2(n_upper_arm_kuka[2], np.sqrt(n_upper_arm_kuka[0]**2 + n_upper_arm_kuka[1]**2))
        joint_commands['left_shoulder_y'] = np.clip(shoulder_pitch, *JOINT_LIMITS['left_shoulder_y'])

        # 4. KUKA Joint 3 (Shoulder Roll) - `lbr_iiwa_joint_3`
        # This joint controls the rotation of the upper arm segment itself (roll along its axis).
        # It's challenging to derive accurately with only shoulder and elbow landmarks without a torso reference.
        # The following calculation is a proxy, using the angle of the upper arm's XZ projection in MediaPipe frame
        # relative to the MediaPipe Z-axis. This might need empirical tuning or a more advanced approach.
        roll_proxy = self._calculate_angle_3d(
            np.array([0, 0, 1]),  # MediaPipe Z-axis (upward in MP's coordinate system)
            np.zeros(3),          # Origin (shoulder)
            np.array([n_upper_arm[0], 0, n_upper_arm[2]]) # MP XZ projection of upper arm
        )
        # Adjust sign based on MediaPipe's X-axis direction
        if n_upper_arm[0] < 0: 
             roll_proxy = -roll_proxy
        joint_commands['left_shoulder_x'] = np.clip(roll_proxy, *JOINT_LIMITS['left_shoulder_x'])


        # 5. KUKA Joint 5 (Forearm Twist) - `lbr_iiwa_joint_5`
        # Rotation of the forearm about the upper arm axis.
        # This calculation uses vectors in the original MediaPipe frame (n_upper_arm, n_forearm)
        # as it's a relative angle between segments.
        axis_twist = np.cross(n_upper_arm, n_forearm)
        if np.linalg.norm(axis_twist) < 1e-6: # Vectors are nearly parallel, twist is undefined or 0
            forearm_twist = 0.0
        else:
            # proj_twist is a vector perpendicular to n_upper_arm and axis_twist.
            # It represents a "zero" orientation for the twist in the plane of rotation.
            proj_twist = np.cross(n_upper_arm, axis_twist)
            forearm_twist = self._calculate_angle_3d(n_forearm, np.zeros(3), proj_twist)
            # Determine the sign of the twist. This heuristic needs testing with your robot.
            if np.dot(n_forearm, axis_twist) < 0: 
                forearm_twist = -forearm_twist
        joint_commands['left_forearm_twist'] = np.clip(forearm_twist, *JOINT_LIMITS['left_forearm_twist'])

        # 6. KUKA Joint 6 (Wrist Pitch) - `lbr_iiwa_joint_6`
        # Pitch (flexion/extension) of the wrist relative to the forearm.
        # This is derived from the pitch angle of the transformed forearm vector (`n_forearm_kuka`)
        # with respect to the horizontal plane in the KUKA base frame.
        wrist_pitch_val = np.arctan2(n_forearm_kuka[2], np.sqrt(n_forearm_kuka[0]**2 + n_forearm_kuka[1]**2))
        joint_commands['left_wrist_pitch'] = np.clip(wrist_pitch_val, *JOINT_LIMITS['left_wrist_pitch'])

        # 7. KUKA Joint 7 (Wrist Yaw) - `lbr_iiwa_joint_7`
        # Yaw (radial/ulnar deviation) of the wrist relative to the forearm.
        # Derived from the yaw angle of the transformed forearm vector (`n_forearm_kuka`)
        # with respect to the X-axis in the KUKA base frame.
        wrist_yaw_val = np.arctan2(n_forearm_kuka[1], n_forearm_kuka[0])
        joint_commands['left_wrist_yaw'] = np.clip(wrist_yaw_val, *JOINT_LIMITS['left_wrist_yaw'])

        logger.debug(f"Computed joint_commands: {joint_commands}")
        return joint_commands

    def smooth_joint_commands(self, current_joint_commands):
        """
        Applies a simple moving average filter to joint commands to reduce noise.

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
        Applies a simple proportional feedback (P-controller).

        Args:
            target_commands (dict): Dictionary of desired joint names and angles (from smoothing).
            actual_joint_positions (dict): Dictionary of actual joint names and their current angles
                                           (e.g., from PyBullet simulation).

        Returns:
            dict: Adjusted target commands.
        """
        adjusted_commands = target_commands.copy()
        for joint_name, target_angle in target_commands.items():
            if joint_name in actual_joint_positions:
                actual_angle = actual_joint_positions[joint_name]
                error = target_angle - actual_angle
                adjusted_angle = target_angle + self.feedback_gain * error
                
                # Apply joint limits to the adjusted angle
                if joint_name in JOINT_LIMITS:
                    adjusted_angle = np.clip(adjusted_angle, *JOINT_LIMITS[joint_name])
                
                adjusted_commands[joint_name] = adjusted_angle
            else: 
                logger.warning(f"Actual position for joint {joint_name} not available for feedback. Using target angle.")
                
        return adjusted_commands
    
    def map_and_filter_commands(self, smoothed_commands: dict, actual_joint_positions: dict) -> dict:
        """
        Maps human-readable joint names in `smoothed_commands` to robot joint names
        (using `NAME_MAP`) and filters to only those joints that are present in
        `actual_joint_positions` (i.e., actually exist on the robot/simulation).

        Args:
            smoothed_commands (dict): Human joint command values (e.g., from `smooth_joint_commands`).
            actual_joint_positions (dict): Dictionary of valid robot joint names and their current positions.

        Returns:
            dict: Filtered and renamed command dictionary ready for robot control.
        """
        filtered = {}
        for cmd_name, value in smoothed_commands.items():
            robot_name = NAME_MAP.get(cmd_name)
            if robot_name and robot_name in actual_joint_positions:
                filtered[robot_name] = value
            else:
                logger.debug(f"No valid robot joint mapped or found for '{cmd_name}'. Skipping.")
        return filtered

if __name__ == '__main__':
    # This block provides example usage and is for informational purposes.
    # It demonstrates how the JointTrajectoryGenerator would typically be used
    # within a larger system involving a vision module and a simulation module.

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
    logger.info("            # NOTE: You need to implement sim.get_joint_states() in PyBulletSimulation")
    logger.info("            # It should return a dictionary like {joint_name: (position, velocity, reaction_force, motor_torque)}")
    logger.info("            actual_joint_states = sim.get_joint_states() ")
    logger.info("            actual_joint_positions = {name: state[0] for name, state in actual_joint_states.items()}")

    logger.info("            # Adjust commands with feedback")
    logger.info("            final_commands = generator.adjust_commands_with_feedback(smoothed_commands, actual_joint_positions)")

    logger.info("            # Map to robot-specific names and filter for valid joints")
    logger.info("            robot_commands = generator.map_and_filter_commands(final_commands, actual_joint_positions)")

    logger.info("            sim.set_joint_positions(robot_commands)")
    logger.info("            sim.step_simulation()")
    logger.info("            # ... add time.sleep if needed ...")

