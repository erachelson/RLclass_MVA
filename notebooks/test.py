import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
# use render_mode="human" to open the game window
env = gym.make('FrozenLake-v1', render_mode="human")
env.reset()
print(env.render())
env.render()

# use render_mode="ansi" to print the game in the console
from time import sleep
sleep(5)