# RL class @ MVA

Welcome to the website of the reinforcement learning class for the [MVA master program](https://www.master-mva.com/).  
This class is joint work between [Emmanuel Rachelson](https://people.isae-supaero.fr/emmanuel-rachelson) (6 sessions) and [Claire Vernade](https://www.cvernade.com/) (2 sessions).  
The class is complemented with an invited lecture after the first 8 sessions.

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

Chapters 0 to 3 cover the foundational ingredients of reinforcement learning, and introduce the three fundamental challenges in RL which are explored independently in the remainder of the class:
- approximation and representation of value functions and policies (a bit of chapter 3, and chapter 4)
- the policy optimization problem (chapters 5 and 6)
- the exploration / exploitation tradeoff (chapters 7 and 8)
Hence, the class is built as a three-branches tree. It will always start with the "stem" of chapters 0 to 3. But later sessions might interleave chapters of the three branches.   
For the sake of brevity and ease of notation, in the schedule below, chapters 0 to 3 are called "foundations", chapter 4 is called "approximation", chapters 5 and 6 are called "optimization", and chapters 7 and 8 are called "exploration".  

## Class material

Notebooks for the first 6 chapters are accessible at [https://github.com/erachelson/RLclass_MVA](https://github.com/erachelson/RLclass_MVA). Please download the latest version before class.  
Material for the last 2 chapters are accessible at [https://emiliekaufmann.github.io/teaching.html](https://emiliekaufmann.github.io/teaching.html). 

## Schedule for 2024-25

The schedule is designed around 3-hours sessions. It might be adjusted depending on the progression of classes. Each session will cover a chapter (session 1 covers chapters 0 and 1).  
Tentative schedule (possible adjustments forthcoming):   
Oct 07 - chapters 0 and 1 (foundations 1/3)  
Oct 14 - chapter 2 (foundations 2/3)  
Oct 21 - chapter 3 (foudnations 3/3)  
Nov 4 - chapter 7 (exploration 1/2)  
Nov 13 - chapter 4 (approximation 1/1)  
Nov 18 - chapter 8 (exploration 2/2)  
Nov 25 - chapter 5 (optimization 1/2)  
Dec 2 - chapter 6 (optimization 2/2)  
Dec 9 invited lecture by Dr. [Vincent François-Lavet](http://vincent.francois-l.be/)  

All classes are scheduled from 9am to 12pm. Please arrive early so we can start on time.  
All class dates are Mondays, except for Nov 13 which is a Wednesday.  
All classes are given at the "Pavillon 3" amphitheater, on the [Site des Cordeliers, Université Paris Cité](https://maps.app.goo.gl/SAy9CZzFbkud3Gmi9), except for the class on Nov 18 that will be given at Amphi Dieulafoy of [Hopital Cochin, Université Paris Cité](https://maps.app.goo.gl/PL48qXXus6NSNGC16).

## Homework

Each notebook contains homework that help play with the concepts introduced in class, to better grasp them. Most exercises come with solutions. The homework also introduces additional important notions. They are a full and important part to reach the class goals. Often, the provided answer reaches out further than the plain question asked and provides comments, additional insights, or external references.

## Evaluation

The final grade will be composed of three parts (coefficients TBD).  
1. Between session 2 and session 8 (included), a short mandatory online 10-15 minutes quiz will be run at the beginning of class, on the contents of the previous session. These quizes will be graded and will count towards the final grade.  
2. An implementation project (topic disclosed around session 8) will also be graded.  
3. An independent assignment on the exploration part will finally be graded.  

