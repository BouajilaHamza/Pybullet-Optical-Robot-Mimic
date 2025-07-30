import cv2
import time
import numpy as np

from vision_module import capture_and_preprocess_frame, estimate_pose, draw_landmarks
from control_module import JointTrajectoryGenerator, JOINT_LIMITS
from simulation_module import PyBulletSimulation

def main():
    # Initialize modules
    sim = PyBulletSimulation(gui=True)
    generator = JointTrajectoryGenerator(smoothing_window_size=5, feedback_gain=0.1)

    # Load humanoid robot in PyBullet
    # IMPORTANT: Replace 'humanoid/humanoid.urdf' with the actual path to your humanoid URDF
    # You might need to adjust the start_pos based on your URDF's scale
    try:
        sim.load_robot("humanoid/humanoid.urdf", start_pos=[0, 0, 1.5])
    except Exception as e:
        print(f"Could not load humanoid/humanoid.urdf: {e}")
        print("Attempting to load a simpler KUKA arm model for demonstration.")
        try:
            sim.load_robot("kuka_iiwa/model.urdf", start_pos=[0, 0, 0.5])
        except Exception as e_kuka:
            print(f"Could not load kuka_iiwa/model.urdf: {e_kuka}")
            print("Exiting as no robot could be loaded.")
            sim.disconnect()
            return

    # Check if a robot was successfully loaded
    if sim.robotId is None:
        print("No robot loaded. Exiting.")
        sim.disconnect()
        return

    # Main teleoperation loop
    cap = cv2.VideoCapture(0) # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open camera. Please ensure a webcam is connected and available.")
        sim.disconnect()
        return

    print("Starting teleoperation. Press 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 1. Vision Module: Capture, Preprocess, and Estimate Pose
            # MediaPipe expects RGB, but OpenCV reads BGR. estimate_pose handles conversion.
            display_frame = cv2.resize(frame, (640, 480)) # Resize for display consistency
            landmarks = estimate_pose(display_frame)

            target_commands = {}
            if landmarks:
                # 2. Control Module: Map landmarks to joint commands and smooth
                raw_commands = generator.map_landmarks_to_joint_commands(landmarks)
                smoothed_commands = generator.smooth_joint_commands(raw_commands)

                # 3. Feedback Loop: Get actual joint positions and adjust commands
                actual_joint_states = sim.get_joint_states()
                actual_joint_positions = {name: state[0] for name, state in actual_joint_states.items()}

                # Filter out joints not present in the loaded robot's actual_joint_positions
                filtered_smoothed_commands = {k: v for k, v in smoothed_commands.items() if k in actual_joint_positions}

                final_commands = generator.adjust_commands_with_feedback(filtered_smoothed_commands, actual_joint_positions)

                # 4. Simulation Module: Apply commands and step simulation
                sim.set_joint_positions(final_commands)

                # Draw landmarks on the display frame
                draw_landmarks(display_frame, landmarks)

            sim.step_simulation()
            time.sleep(1./240.) # Maintain real-time speed if possible (PyBullet default time step)

            cv2.imshow('Teleoperation View', display_frame)

            if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to exit
                break

    except Exception as e:
        print(f"An error occurred during teleoperation: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sim.disconnect()
        print("Teleoperation ended.")

if __name__ == '__main__':
    main()


