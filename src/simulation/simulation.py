#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import pybullet as p
import pybullet_data
import os
import math
import numpy as np
from utils import current_joint_positions, get_point_cloud
import h5py
import time
import tf
import logging
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tensorflow
import cv2


#Add object
#Report + GitHub
#Implement object-level encoding



# Configuration

NUM_SIMS = 10
SIM_RESULTS = []
ARM_JOINTS = [2, 3, 4, 5, 6, 7]
FINGER_JOINTS = [9, 10, 11, 12, 13, 14]
HOME_POS = [math.pi] * len(ARM_JOINTS)
MAX_FINGER_POS = 1.5

COLOR = "green"

STATE_DURATIONS = [3.0, 3.0, 1.0, 3.0, 3.0]#, 1.0]
CONTROL_DT = 1. / 960.

initial_position = None
target_position = None



model_dir = 'Windows/media/neel/Windows/Users/kulka/generalizable_task_planning/src/simulation/mask_rcnn_inception_v2_coco.config'

def load_model():
    model = tensorflow.saved_model.load(model_dir)
    return model

def run_inference_for_single_image(model, image):
    image_np = np.asarray(image)
    input_tensor = tensorflow.convert_to_tensor(image_np)
    input_tensor = input_tensor[tensorflow.newaxis, ...]
    
    output_dict = model(input_tensor)
    
    
    output_dict = {key:value.numpy() for key,value in output_dict.items()}
    
    num_detections = int(output_dict['num_detections'][0])
    boxes = output_dict['detection_boxes'][0]
    classes = output_dict['detection_classes'][0]
    scores = output_dict['detection_scores'][0]
    
    return boxes, classes, scores, num_detections



#Object segmentation

def get_rgbd_images():
    _, rgb_image = p.getCameraImage(width=640, height=480)
    rgb_image = np.array(rgb_image)

    _, _, depth_image = p.getCameraImage(width=640, height=480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_image = np.array(depth_image)

    return rgb_image, depth_image

def segment_objects_in_environment(model):
    rgb_image, depth_image = get_rgbd_images()
    boxes, classes, scores, num_detections = run_inference_for_single_image(model, rgb_image)

    for i in range(num_detections):
        if scores[i] > 0.5:  
            box = boxes[i]
            class_id = int(classes[i])
            cv2.rectangle(rgb_image, 
                        (int(box[1] * rgb_image.shape[1]), int(box[0] * rgb_image.shape[0])),
                        (int(box[3] * rgb_image.shape[1]), int(box[2] * rgb_image.shape[0])), 
                        (0, 255, 0), 2)
        
    cv2.imshow('Segmented Image', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def setup_environment(with_gui):
    """
    Set up the PyBullet environment by loading the Kinova robot and a table.
    """
    urdf_root_path = pybullet_data.getDataPath()
    client_id = -1
    if with_gui:
        client_id = p.connect(p.GUI)
    else:
        client_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(urdf_root_path)
    p.resetDebugVisualizerCamera(3, 90, -30, [0.0, -0.0, -0.0])
    table_uid = p.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
    kinova_uid = p.loadURDF("./robot/j2s6s300.urdf", useFixedBase=True)

    
    num_joints = p.getNumJoints(kinova_uid)
    for i in range(num_joints):
        p.changeDynamics(kinova_uid, i, mass=5.0)

    plank_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.01, 0.3])
    plank_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.01, 0.3], rgbaColor=[0.5, 0.5, 0.5, 1])
    plank_uid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plank_shape, baseVisualShapeIndex=plank_visual_shape, basePosition=[0.5, 0, 0.05])


    return kinova_uid, table_uid, plank_uid, client_id

def initialize_robot_position(kinova_uid):
    """
    Initialize the robot to the home position.
    """
    home_state_dict = dict(zip(ARM_JOINTS, HOME_POS))
    for joint, pos in home_state_dict.items():
        p.resetJointState(kinova_uid, joint, pos)

def reset_to_initial_state(move_group, initial_end_effector_position):

    move_group.set_joint_value_target(initial_end_effector_position)
    move_group.go(wait=True)

    move_group.stop()
    move_group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

    

def create_object():
    """
    Create a block object in the environment.
    """

    block_uids = []
    colors = [
        [1, 0, 0, 1],  # Red
        [0, 0, 1, 1],  # Blue
        [0, 1, 0, 1]   # Green
    ]

    x_min, x_max = 0.0, 0.6
    y_min, y_max = -0.3, -0.05

    for color in colors:
        block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        block_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=color)
        block_body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=block_shape, baseVisualShapeIndex=block_visual_shape)
        p.changeDynamics(block_body, -1, lateralFriction=4.5, spinningFriction=1.0, rollingFriction=1.0)

        initial_x = np.random.uniform(x_min, x_max)
        initial_y = np.random.uniform(y_min, y_max)
        initial_position = [initial_x, initial_y, 0.0]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(block_body, initial_position, initial_orientation)

        block_uids.append(block_body)

    return block_uids

def run_simulation(kinova_uid, object_uid, client_id, move_group):
    """
    Simulation loop that runs for a fixed number of states.
    """

    js_list = []
    current_state = 0
    state_time = 0.
    with h5py.File("sim_results.h5", "w") as hdf_file:

        joint_states_dataset = hdf_file.create_dataset('joint_states', shape=(0, len(ARM_JOINTS)), maxshape=(None, len(ARM_JOINTS)), dtype='f')

        for state in STATE_DURATIONS:
            
            
                    
            joint_positions = current_joint_positions(kinova_uid, ARM_JOINTS)
            js_list.append(joint_positions)

            joint_states_dataset.resize(joint_states_dataset.shape[0] + 1, axis=0)
            joint_states_dataset[-1, :] = joint_positions
            
                
            control_robot_state(kinova_uid, object_uid,current_state, move_group)
            current_state += 1
        
        SIM_RESULTS.append(js_list)


def arm_control(object_uid, color, move_group, state):
    target_position = list(p.getBasePositionAndOrientation(object_uid[color])[0])
    if (state == 0): 
        target_position[2] += 0.1
    elif (state == 1):
        target_position[2] += 0.03
    elif (state == 3):
        target_position[2] += 0.2
    elif (state == 4):
        stack_position = [0.5, 0.3, 0.2]
        target_position = stack_position

    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = target_position[0]
    pose_target.position.y = target_position[1]
    pose_target.position.z = target_position[2]
    roll = 3.2
    pitch = 0
    yaw = 0
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    pose_target.orientation.x = quaternion[0]
    pose_target.orientation.y = quaternion[1]
    pose_target.orientation.z = quaternion[2]
    pose_target.orientation.w = quaternion[3]
    

    return plan_and_execute_motion(pose_target, move_group) 
    


def control_robot_state(kinova_uid, object_uid, state, move_group):
    colors = ["red", "blue", "green"]
    color = colors.index(COLOR)

    num_steps = int(STATE_DURATIONS[state] / CONTROL_DT)
    approach_offset = 0.01

    print("state " + str(state))

    if (state != 2):
        trajectory = arm_control(object_uid, color, move_group, state)
        
        if state == 1:
            for i, joint in enumerate(FINGER_JOINTS):
                target_pos =  0
                p.setJointMotorControl2(kinova_uid, joint, p.POSITION_CONTROL, target_pos, force=30)
                p.stepSimulation()

        for i in range(num_steps):
            current_time = i * CONTROL_DT
            index = int(current_time/(STATE_DURATIONS[state]/len(trajectory)))
            joint_positions = trajectory[index]
            #for joint_index in range(p.getNumJoints(kinova_uid)):
            for joint_index in ARM_JOINTS:
                p.setJointMotorControl2(bodyIndex=kinova_uid,jointIndex=joint_index,controlMode=p.POSITION_CONTROL,targetPosition=joint_positions[joint_index-2])
            p.stepSimulation()
            time.sleep(CONTROL_DT)

    elif state == 2: #Close gripper
        for i in range(num_steps):
            for i, joint in enumerate(FINGER_JOINTS):
                target_pos = 0.6 * MAX_FINGER_POS if i % 2 == 0 else 0.5 * MAX_FINGER_POS
                p.setJointMotorControl2(kinova_uid, joint, p.POSITION_CONTROL, target_pos, force=60)
            p.stepSimulation()
            time.sleep(CONTROL_DT)
                            
    

def plan_and_execute_motion(target_position, move_group):

    rospy.loginfo("Setting target pose: {}".format(target_position))

    move_group.set_pose_target(target_position)
    
    plan = move_group.plan()
    move_group.go(wait=True)

    
    trajectory_var = plan[1]
    trajectory_var = trajectory_var.joint_trajectory.points
    
    trajectory = [point.positions for point in trajectory_var]
    


    move_group.stop()
    move_group.clear_pose_targets()

    return trajectory
    

def simulate(num_sims, with_gui=False):

    model = load_model()
    segment_objects_in_environment(model)


    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_pybullet_integration', anonymous=True)


    moveit_logger = logging.getLogger('moveit_commander')
    moveit_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    moveit_logger.addHandler(ch)


    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("arm")
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

    move_group.set_planner_id("RRTConnectkConfigDefault")


    NUM_SIMS = num_sims

    initial_end_effector_position = move_group.get_current_joint_values()

    for _ in range(NUM_SIMS):
        kinova_uid, _, plank_uid, client_id = setup_environment(with_gui)

        plank_pose = geometry_msgs.msg.PoseStamped()
        plank_pose.header.frame_id = robot.get_planning_frame()
        plank_pose.pose.position.x = 0.5
        plank_pose.pose.position.y = 0.0
        plank_pose.pose.position.z = 0.05
        plank_pose.pose.orientation.w = 1.0
        scene.add_box("plank", plank_pose, size=(0.6, 0.02, 0.6))


        initialize_robot_position(kinova_uid)
        object_uid = create_object()
        run_simulation(kinova_uid, object_uid, client_id, move_group)
        reset_to_initial_state(move_group, initial_end_effector_position)
        p.disconnect()



        