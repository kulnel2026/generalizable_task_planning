import pybullet as p

def current_joint_positions(robot_id, joint_indices):
    """
    
    """
    current_joint_positions = []
    for joint_index in joint_indices:
        joint_state = p.getJointState(robot_id, joint_index)
        current_joint_positions.append(joint_state[0])  # Joint position is the first element of the tuple returned by getJointState
    
    return current_joint_positions