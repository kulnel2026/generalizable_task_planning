import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import pdb

def current_joint_positions(robot_id, joint_indices):
    """
    This function returns the current joint positions of a robot.
    
    Parameters:
    robot_id (int): The unique identifier for the robot.
    joint_indices (list): A list of indices representing the joints of the robot.
    
    Returns:
    current_joint_positions (list): A list of current positions for each joint specified by joint_indices.
    """
    current_joint_positions = []
    for joint_index in joint_indices:
        # getJointState returns a tuple where the first element is the current position of the joint
        joint_state = p.getJointState(robot_id, joint_index)
        current_joint_positions.append(joint_state[0])  # Append the joint position to the list
    
    return current_joint_positions


def getDepth(z_n, zNear, zFar):
    z_n = 2.0 * z_n - 1.0
    z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
    return z_e

def get_point_cloud():
    """
    Generate a point cloud from the depth image.
    """
    cam_pos = [0, 0, 2]
    cam_target = [0, 0, 0]
    cam_up = [0, 1, 0]
    view_matrix = p.computeViewMatrix(
        cameraEyePosition = cam_pos,
        cameraTargetPosition = cam_target,
        cameraUpVector=cam_up
    )
    
    # Define projection matrix
    near_val = 0.01
    far_val = 5.1
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=30.0,
        aspect=1.0,
        nearVal=near_val,
        farVal=far_val
    )
    
    # Get an image
    imgW, imgH, rgb_img, depth_img, segImg = p.getCameraImage(
        width=128, 
        height=128,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    pdb.set_trace()
    
    print("rgbBuffer.shape=", rgb_img.shape)
    print("depthBuffer.shape=", depth_img.shape)
    plt.imshow(depth_img)
    
    stepX = 1
    stepY = 1
    realDepthImg = depth_img.copy()
    for w in range(0, imgW, stepX):
        for h in range(0, imgH, stepY):
            realDepthImg[w][h] = getDepth(depth_img[w][h],near_val,far_val)
            
    plt.imshow(realDepthImg)
    
    # Convert depth image to point cloud
    stepX = 1
    stepY = 1        
    pointCloud = np.empty([np.int32(imgH/stepY), np.int32(imgW/stepX), 4])

    projectionMatrix = np.asarray(projectionMatrix).reshape([4,4],order='F')

    viewMatrix = np.asarray(viewMatrix).reshape([4,4],order='F')

    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

    for h in range(0, imgH, stepY):
        for w in range(0, imgW, stepX):
                x = (2*w - imgW)/imgW
                y = -(2*h - imgH)/imgH
                z = 2*depth_img[h,w] - 1
                pixPos = np.asarray([x, y, z, 1])
                position = np.matmul(tran_pix_world, pixPos)
                pointCloud[np.int32(h/stepY), np.int32(w/stepX), :] = position / position[3]
                
    print(pointCloud.shape)
    
    # Creating 3D figure
    xs = []
    ys = []
    zs = []

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax = plt.axes(projection='3d')

    for h in range(0, imgH, stepY):
        for w in range(0, imgW, stepX):
            [x,y,z,_] = pointCloud[h,w]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            

    ax.scatter(xs, ys, zs, c=zs, marker='.')
    ax.view_init(75, 45)
    plt.draw()

    
    
