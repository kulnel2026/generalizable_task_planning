# Generalizable Task Planning Model
Welcome to the repository for our Generalized Task Planning Robot project. This project is focused on developing a robotic system capable of performing complex task planning, specifically stacking cubes of different colors in the presence of obstacles, using advanced motion planning algorithms and simulation environments.

## Overview
This project integrates multiple tools and libraries to create a comprehensive robotic simulation environment. The robot is tasked with stacking cubes of various colors while navigating around multiple obstacles. The simulation is built using PyBullet for physics and collision detection, MoveIt for motion planning, and ROS for robot control.

## Features
- **Motion Planning With MoveIt**: Utilizes MoveIt for advanced motion planning and execution, supporting various planning algorithms such as RRT-Connect.
- **PyBullet Integration**: Physics simulation and collision detection are handled by PyBullet, providing a realistic environment for testing.
- **Generalizable Task Planning**: Utilizes object-level representations extracted from RGB-D data for planning multi-step manipulation tasks, inspired by "Generizable Planning through Representation Pretraining".
- **Obstacle Avoidance**: Multiple obstacles are dynamically added to the environment, challenging the robot to navigate and stack cubes efficiently.
- **Sim-to-Real Transfer**: Engineered for effective transfer from simulation to reality, ensuring that robotic tasks learned in a virtual environment can be performed seamlessly in actual settings.
- **Plug-and-Play Skill Library**: Facilitates a modular system where new skills can be incorporated into the library and utilized in task planning without the need for extensive retraining.

## Architecture Overview
The project is structured into several key concepts:
1. **Simulation Environment**:
* **PyBullet**: Sets up the environment, including the table, Kinova robot, and obstacles. Handles physics simulation and collision detection.
* **MoveIt**: Manages motion planning and execution, interfacing with ROS to control the Kinova robot.
2. **Robot Control**:
* **Joint Control**: Manages the robot's arm and gripper movements using ROS and MoveIt.
* **State Control**: Controls the robot's behavior through different states, such as approaching, picking, and stacking cubes.
3. **Obstacle Management**:
* **Dynamic Obstacles**: Adds multiple obstacles, to challenge the robot's navigation and planning capabilities.
4. **Image Processing**:
* **Camera Setup**: Configures multiple simulated cameras to capture RGB-D images from different perspectives.
* **Image Processing**: Processes captured images for potential future object detection and scene understanding tasks.

## Getting Started
**Prerequisites**
* Operating System: Ubuntu version: 20.04 LTS 
      * https://releases.ubuntu.com/focal/
* Robot Operating System (ROS): ROS Noetic
      * http://wiki.ros.org/noetic/Installation       
* MoveIt: A motion planning framework for ROS
      * https://github.com/Kinovarobotics/kinova-ros/tree/noetic-devel/kinova_moveit
* Python: Programming language for scripting and running simulations 
* PyBullet: Physics simulation library for robotics
* Additional Libraries: 
       * NumPy: For numerical computations
       * h5py: For handling HDF5 files 
       * geometry_msgs: For defining geometric messages in ROS
       * tf: For handling transformations in ROS 
       * time: For simulation time 

## Future Work
* **Object Detection and Segmentation**: Integrate TensorFlow for real-time object detection and segmentation to enhance task planning and execution.
* **Enhanced Obstacle Avoidance**: Implement more sophisticated algorithms for dynamic obstacle detection and avoidance.

## Citation
```
@misc{wang2022generalizable,
      title={Generalizable Task Planning through Representation Pretraining}, 
      author={Chen Wang and Danfei Xu and Li Fei-Fei},
      year={2022},
      eprint={2205.07993},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
```
@misc{mukherjee2021reactive,
      title={Reactive Long Horizon Task Execution via Visual Skill and Precondition Models}, 
      author={Shohin Mukherjee and Chris Paxton and Arsalan Mousavian and Adam Fishman and Maxim Likhachev and Dieter Fox},
      year={2021},
      eprint={2011.08694},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
# generalizing_task_planning
