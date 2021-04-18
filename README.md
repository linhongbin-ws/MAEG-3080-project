
## MAEG-3080-project

This is the code repository for MAEG-3080 course project, including tutorials, codes for project.

**Overview:**
* [Project overview](https://github.com/linhongbin-ws/MAEG-3080-project/blob/master/project-overview.md)

**Tutorial links:** (_Note: use the following links for preview. If some of them fail to render, try to use the other link_)

* 00-Setting Up IDE ([github link](https://github.com/linhongbin-ws/MAEG-3080-project/blob/master/00-Setting-Up-IDE.md))
* 01-Pytorch Tutorial ([jupyter link](https://nbviewer.jupyter.org/github/linhongbin-ws/MAEG-3080-project/blob/master/01-pytorch-tutorial.ipynb)) ([colab link](https://colab.research.google.com/github/linhongbin-ws/MAEG-3080-project/blob/master/01-pytorch-tutorial.ipynb))
* 02-Monte Carlo Tree Search Tutorial ([jupyter link](https://nbviewer.jupyter.org/github/linhongbin-ws/MAEG-3080-project/blob/master/02-MCTS-tutorial.ipynb)) ([colab link](https://colab.research.google.com/github/linhongbin-ws/MAEG-3080-project/blob/master/02-MCTS-tutorial.ipynb))
* 03-Alpha Zero for Gomoku Tutorial ([jupyter viewer](https://nbviewer.jupyter.org/github/linhongbin-ws/MAEG-3080-project/blob/master/03-Alpha%20Zero%20for%20Gomoku.pdf))

**Project code:**

Students are required to program the following scripts:
- [mcts_alphaZero.py](https://github.com/linhongbin-ws/MAEG-3080-project/blob/master/mcts_alphaZero.py)
- [policy_value_net_pytorch.py](https://github.com/linhongbin-ws/MAEG-3080-project/blob/master/policy_value_net_pytorch.py)
- [train.py](https://github.com/linhongbin-ws/MAEG-3080-project/blob/master/train.py)

The scripts will try to finish
- Program a Network for alpha zero using Pytorch
- Program a Monte Carlo Tree Search for alpha zero.
- Program the training pipeline
- Visualize the evaluation of the trained model against a pure Monte Carlo Tree Search model. 

[Note]:
- Try to program without copying classmate's code or other online code.

[comment]: <> (## Resource)

[comment]: <> (Monte Carlos Tree Search &#40;MCTS&#41;:)

[comment]: <> (1. Udacity RL Course Series: [link]&#40;https://classroom.udacity.com/courses/ud600/lessons/4759058600/concepts/51904801690923&#41; )

[comment]: <> (1. Clear Explaination by John Levine: [link]&#40;https://www.youtube.com/watch?v=UXW2yZndl7U&t=2s&ab_channel=JohnLevine&#41;)

[comment]: <> (1. Chinese Course on MCTS: [link]&#40;https://www.youtube.com/watch?v=niIaKaWIRX0&ab_channel=%E4%B8%AD%E5%9B%BD%E5%A4%A7%E5%AD%A6MOOC-%E6%85%95%E8%AF%BE&#41;)

[comment]: <> (1. Survey Paper for MCTS &#40;Deep Understanding on MCTS and its variations&#41;: [link]&#40;https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6145622&casa_token=AecTrST5MJYAAAAA:1UepYH0lA9-jdodOaItjidj0ie8kcKFAH65qh4F3AzkX1wiWrfNj4lb5Um-w7RJChEu0heo3&tag=1&#41;)

[comment]: <> (1. Pytorch Tutorial [link]&#40;https://github.com/yunjey/pytorch-tutorial&#41;.)