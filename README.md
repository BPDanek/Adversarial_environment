

####Using Deep Q-Network to Learn How To Play Flappy Bird

<img src="./images/demo_2.mov" width="250">
See images/demo_1, and images/demo_2 for a demo of how this works. 

####Overview
This project takes a fully trained reinforcement learning agent [borrowed from here](https://github.com/yenchenlin/DeepLearningFlappyBird), and produces an adversarial example which (when fed back to the agent) draws an action of our choosing.

####Installation Dependencies:
* Python 3
* TensorFlow 1.4
* pygame
* OpenCV-Python

#### How to Run?
```
git clone https://github.com/BPDanek/Adversarial_environment.git
cd Adversarial_environment
python DEMO_deep_q_network.py
```
#### How do Adversarial Examples Work?
Adversarial Examples are additions to the input data of a neural network, which causes a specific (often incorrect) 
output. This is not a result of insufficient training, or some sort of evidence of bad neural network architecture
(although solutions to adversarial examples exist in this space), rather they are evidence that the current state of
the art of neural networks is un-robust, and fails to capture truly important features from the input data. There is a 
remarkable amount of research done on this concept, as it is a potentially catastrophic fault for neural networks used
for controlling something on the order of cars, or a robot.
  
The process of creating an adversarial example treats a neural network as a large function, which takes some data in, 
and produces some output. Just like the loss function of a DL program is responsible for directing the gradient steps 
in an optimizer, an adversarial objective function is optimized to meet the objective of the adversary. In our case, 
the objective is to maximize the Q-Value for a specific action (flap the bird) by modifying the input data with noise.
Our objective is unconstrained, and thus can be solved in a few gradient steps (but is totally visible by the 
naked eye, something other papers have been able to get around  as well).  We do, however make the attack binary so that
it cannot be protected by a basic thresholding operation.

The science behind binarizing the attack follows [1]'s method of creating "checkerboard" structures in bones by adding 
a dynamic nonlinear activation to the input data of the network. The heuristic is that initially, the activation is
dormant, but as the number of optimization steps increases, the nonlinearity becomes more nonlinear, to ultimately 
approach a binary threshold function (signum about some center, in our case 255/2). 

This research is unique in the sense that it shows reinforcement learning agents are susceptible to the weaknesses CNN's
are so famous for. The value in this is to uncover flaws in neural network schemes (particularly  ones used to make 
decisions things) that may cause harm. 

#### Experiments

There were countless experiments, but the relevant structure of the program is as such:
```deep_q_network.py``` is the original version of the agent
```DEMO_deep_q_network.py``` is the demo of the adversarial example working (press 1 to insert perturbation live)
```deep_q_network_v2_2.py``` is the site for manufacturing the adversarial example (removed)

(There will be more documents uploaded to support this in the extremely near future)

##### Optimization Scheme

(this section will be explained in the future/tbd)

##### Training

(reproducing this work will also be revealed in a future upload)

#### References

[1] Infill Optimization for Additive Manufacturing—Approaching Bone-Like Porous Structures by Jun Wu, Niels Aage, Rudiger Westermann, and Ole Sigmund

#### Disclaimer
This work is highly based on the following repos:

1. [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)
3. [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

