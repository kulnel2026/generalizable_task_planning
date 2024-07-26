# Generalizable Task Planning Model
## Overview
This repository contains an implementation of advanced machine learning models aimed at enabling robots to perform multi-step manipulation tasks in unseen environments. Inspired by recent advances in representation learning and reactive task execution, our models leverage pretraining on large-scale datasets and visual skill acquisition to generalize across different tasks without the need for retraining or fine-tuning.

## Features
- **Generalizable Task Planning**: Utilizes object-level representations extracted from synthetic datasets for planning multi-step manipulation tasks, inspired by "Generizable Planning through Representation Pretraining".
- **Reactive Execution**: Implements a reactive system that dynamically adjusts to execution failures using learned visual skill and precondition models, as detailed in "Reactive Long Horizon Task Execution via Visual Skill and Precondition Models".
- **Sim-to-Real Transfer**: Designed for robust sim-to-real transfer, enabling seamless execution of robotic tasks learned in simulation in real-world settings.
- **Plug-and-Play Skill Library**: Supports a modular approach, allowing new skills to be added to the system's library and used in task planning without extensive retraining.

## Architecture Overview
The architecture integrates two primary components for robotic task execution: a Generalizable Planning Model and a Reactive Execution System. The planning model leverages pre-trained object-level representations to plan multi-step manipulation tasks in unseen environments, drawing on large-scale scene understanding datasets. For reactive execution, the system employs visual skill and precondition models that dynamically adjust to execution discrepancies, enabling real-time adaptation. This dual-component approach ensures robust performance across a variety of tasks and environments, emphasizing the system's capacity for sim-to-real transfer and modular skill integration.

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
