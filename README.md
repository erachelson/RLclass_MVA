# RL class @ MVA

Welcome to the website of the reinforcement learning class for the [MVA master program](https://www.master-mva.com/).  
This class is joint work between [Emmanuel Rachelson](https://people.isae-supaero.fr/emmanuel-rachelson) (first 6 sessions) and [Emilie Kaufmann](https://emiliekaufmann.github.io/) (2 last ones).

## Syllabus

This class aims at providing a comprehensive and modern introduction to reinforcement learning concepts and algorithms. It endeavors to provide a solid formal basis on foundational notions of reinforcement learning (MDP modeling, convergence properties of dynamic programming and stochastic gradient descent, stochastic bandits, etc.), in order to move in a principled manner towards state-of-the-art algorithms (including deep RL ones).  
The class is structured around a series of chapters, each covered in an independent notebook.  

**Chapter 0: Reinforcement Learning class introduction; key intuitions**  
Class rules, general definition of RL, position in the ML landscape, first elements of vocabulary.  
**Chapter 1: Modeling sequential decision problems with Markov Decision Processes**  
MDP definition, policies and value functions, definition of optimality, state distributions, horizon.  
**Chapter 2: Characterizing value functions: the Bellman equations**  
State-action value functions, dynamic programming evaluation and optimality Bellman equations, value iteration, (modified) policy iteration, asynchronous dynamic programming, linear programming.  
**Chapter 3: Learning value functions**  
Approximate value and policy iteration, AVI as a series of supervised learning problems, stochastic gradient descent for AVI, temporal difference methods, Q-learning. Overview of key intrinsic challenges in RL.  
**Chapter 4: From fitted Q-iteration to deep Q-networks**  
fitted Q-iteration, neural network architecture for value functions, DQN, improvements on DQN.  
**Chapter 5: Continuous actions in DQN algorithms**  
From DDPG to SAC.  
**Chapter 6: Direct policy search and policy gradient methods**  
Policy gradient theorem, REINFORCE, A2C, PPO, evolutionary RL.  
**Chapter 7: Stochastic bandits**  
Regret. Explore Then Comit. UCB. Thompson Sampling. Contextual bandits. Bandits beyond RL.  
**Chapter 8: Bandit tools for Reinforcement Learning**  
Exploration in RL. (Bandit based) Monte Carlo Tree Search. UCT, Alpha Zero.  

## Class material

Notebooks for the first 6 chapters are accessible at [https://github.com/erachelson/RLclass_MVA](https://github.com/erachelson/RLclass_MVA). Please download the latest version before class.

## Schedule for 2023-24

The schedule is designed around 3-hours sessions. It might be adjusted depending on the progression of classes.

Session 1: chapters 0 and 1.  
Session 2: chapter 2.  
Session 3: chapter 3.  
Session 4: chapter 4.  
Session 5: chapter 5.  
Session 6: chapter 6.  
Session 7 and 8: stochastic bandits, monte carlo tree search and alphaGo.  

## Homework

Each notebook contains homework that help play with the concepts introduced in class, to better grasp them. Most exercises come with solutions. The homework also introduces additional important notions. They are a full and important part to reach the class goals. Often, the provided answer reaches out further than the plain question asked and provides comments, additional insights, or external references.

## Evaluation

The final grade will be composed of three parts (coefficients TBD).  
1. Between session 2 and session 6 (included), a short mandatory online 10-15 minutes quiz will be run at the beginning of class, on the contents of the previous session. These quizes will be graded and will count towards the final grade.  
2. An implementation project around session 6 will also be graded.  
3. An independent assignment on the last two sessions will finally be graded.  

