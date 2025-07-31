import cv2
import time
import pybullet as pb
from modules.vision_module import  estimate_pose, draw_landmarks
from modules.control_module import JointTrajectoryGenerator
from modules.simulation_module import PyBulletSimulation
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
def main():
    sim = PyBulletSimulation(gui=True)
    generator = JointTrajectoryGenerator(smoothing_window_size=5, feedback_gain=0.1)


    try:
        sim.load_table()
        sim.load_robot("kuka_iiwa/model.urdf", start_pos=[-0.2, 0, 0.6])
        sim.load_object(pos=[-0.7,0,0.8])
        sim.load_plate()
    except Exception as e:
        logger.error(f"Could not load kuka_iiwa/model.urdf: {e}")
        logger.error("Exiting as no robot could be loaded.")
        sim.disconnect()
        return

    if sim.robotId is None:
        logger.error("No robot loaded. Exiting.")
        sim.disconnect()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera. Please ensure a webcam is connected and available.")
        sim.disconnect()
        return

    logger.info("Starting teleoperation. Press 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Ignoring empty camera frame.")
                continue

            display_frame = cv2.resize(frame, (640, 480))
            landmarks = estimate_pose(display_frame)
            logger.debug(f"landmarks {landmarks} \n {"-"*100}")
            if landmarks:
                raw_commands = generator.map_landmarks_to_joint_commands(landmarks)
                logger.debug(f"raw_commands {raw_commands} \n {"-"*100}")
                smoothed_commands = generator.smooth_joint_commands(raw_commands)
                logger.debug(f"smoothed commands {smoothed_commands} \n {"-"*100}")
                state_tuples = pb.getJointStates(sim.robotId, sim.joint_indices.values())
                logger.debug(f"state_tuples {state_tuples} \n {"-"*100}")
                actual_joint_positions = {name: state[0] for name, state in zip(sim.joint_indices.keys(), state_tuples)}
                logger.debug(f"actual joint positions {actual_joint_positions} \n {"-"*100}")
                filtered_smoothed_commands = generator.map_and_filter_commands(smoothed_commands, actual_joint_positions)
                logger.debug(f"filtered smoothed commands {filtered_smoothed_commands} \n {"-"*100}")
                final_commands = generator.adjust_commands_with_feedback(filtered_smoothed_commands, actual_joint_positions)
                logger.debug(f"final commands {final_commands} \n {"-"*100}")

                sim.set_joint_positions(final_commands)
                draw_landmarks(display_frame, landmarks)

            sim.step_simulation()
            time.sleep(1./240.) 

            cv2.imshow('Teleoperation View', display_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    except Exception as e:
        logger.error(f"An error occurred during teleoperation: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sim.disconnect()
        logger.info("Teleoperation ended.")

if __name__ == '__main__':
    main()


