import pybullet as p
import pybullet_data
import time
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class PyBulletSimulation:
    def __init__(self, gui=True):
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT) # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = None
        self.joint_names = []
        self.joint_indices = {}

    def load_robot(self, urdf_path, start_pos=[0, 0, 1], start_orientation=[0, 0, 0, 1]):
        """
        Loads a robot URDF into the simulation.

        Args:
            urdf_path (str): Path to the URDF file.
            start_pos (list): Initial position [x, y, z].
            start_orientation (list): Initial orientation as a quaternion [x, y, z, w].
        """
        self.robotId = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=False)
        logger.info(f"Loaded robot with ID: {self.robotId}")

        # Get joint info
        num_joints = p.getNumJoints(self.robotId)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.joint_names.append(joint_name)
                self.joint_indices[joint_name] = i
                logger.info(f"  Joint {i}: {joint_name} (Type: {joint_type})")

    def load_table(self, start_pos=[0, 0, 0]):
        table_urdf = "table/table.urdf"
        try:
            self.tableId = p.loadURDF(table_urdf, start_pos, useFixedBase=True)
            logger.info(f"Loaded table with ID: {self.tableId}")
        except Exception as e:
            logger.error(f"Failed to load table: {e}")

    def load_plate(self, urdf_path="tray/traybox.urdf", pos=[0.5, 0, 0.65], orientation=[0, 0, 0]):
        try:
            plateId = p.loadURDF(urdf_path, pos, p.getQuaternionFromEuler(orientation), useFixedBase=True)
            logger.info(f"Loaded plate object with ID: {plateId}")
            return plateId
        except Exception as e:
            logger.error(f"Failed to load plate: {e}")
            return None

    def load_object(self, obj_path="duck_vhacd.urdf", pos=[0.9, 0, 0.8], orientation=[0, 0, 0]):
        try:
            object_id = p.loadURDF(obj_path, pos, p.getQuaternionFromEuler(orientation), useFixedBase=False)
            logger.info(f"Loaded object with ID: {object_id}")
            return object_id
        except Exception as e:
            logger.error(f"Failed to load object: {e}")
            return None

    def set_joint_positions(self, joint_commands, kp=0.5, kd=1.0):
        """
        Sets target positions for specified joints.

        Args:
            joint_commands (dict): Dictionary of joint names and their target angles.
            kp (float): Proportional gain for the joint motors.
            kd (float): Derivative gain for the joint motors.
        """
        if self.robotId is None:
            logger.error("Error: No robot loaded.")
            return

        for joint_name, target_angle in joint_commands.items():
            if joint_name in self.joint_indices:
                logger.info(f"joint_name: {joint_name} target_angle: {target_angle}")
                joint_index = self.joint_indices[joint_name]
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=round(target_angle,1),
                    positionGain=kp,
                    velocityGain=kd
                )
            else:
                logger.warning(f"Warning: Joint \'{joint_name}\' not found in robot.")

    def step_simulation(self, time_step=1./240.):
        """
        Advances the simulation by one step.

        Args:
            time_step (float): The time step for the simulation.
        """
        p.setTimeStep(time_step)
        p.stepSimulation()

    def disconnect(self):
        """
        Disconnects from the physics server.
        """
        p.disconnect()

if __name__ == '__main__':
    sim = PyBulletSimulation(gui=True)
    try:
        try:
            sim.load_robot("humanoid/humanoid.urdf", start_pos=[0, 0, 1.5])
        except p.error:
            logger.error("humanoid/humanoid.urdf not found or failed to load. Trying kuka_iiwa/model.urdf...")
            sim.load_robot("kuka_iiwa/model.urdf", start_pos=[0, 0, 0.5])

        if sim.robotId is not None:
            if "lbr_iiwa_joint_1" in sim.joint_indices:
                target_angles = {
                    "lbr_iiwa_joint_1": 0.5,
                    "lbr_iiwa_joint_2": 0.0,
                    "lbr_iiwa_joint_3": 0.0,
                    "lbr_iiwa_joint_4": -1.0,
                    "lbr_iiwa_joint_5": 0.0,
                    "lbr_iiwa_joint_6": 1.0,
                    "lbr_iiwa_joint_7": 0.0
                }
                print("Applying example KUKA arm joint commands...")
                sim.set_joint_positions(target_angles)


            elif "abdomen" in sim.joint_indices:
                target_angles = {
                    "abdomen": 0.2, # Bend forward slightly
                    "right_hip": 0.5, # Lift right leg slightly
                    "left_hip": -0.5 # Lower left leg slightly
                }
                logger.info("Applying example humanoid joint commands...")
                sim.set_joint_positions(target_angles)

            for i in range(240 * 5): # Run for 5 seconds
                sim.step_simulation()
                time.sleep(1./240.) # Maintain real-time speed if possible

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        sim.disconnect()



    def get_joint_states(self):
        """
        Gets the current joint states (position, velocity, etc.) of the loaded robot.

        Returns:
            dict: Dictionary of joint names and their states (position, velocity, reaction forces, applied torque).
        """
        if self.robotId is None:
            logger.error("Error: No robot loaded.")
            return {}

        joint_states = {}
        for joint_name, joint_index in self.joint_indices.items():
            joint_state = p.getJointState(self.robotId, joint_index)
            # joint_state is a tuple: (position, velocity, reaction_forces, applied_torque)
            joint_states[joint_name] = joint_state
        return joint_states