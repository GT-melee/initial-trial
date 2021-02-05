import time
from time import sleep
import tkinter
import matplotlib

from charlieenv import CharlieEnv, GetCharlieEnvClass
from evaluatebob import evaluate
from vectorizeenv import VectorizedClass

matplotlib.use('TkAgg')
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import *
from stable_baselines3 import PPO
from bobenv import BobEnv, GetBobEnvClass


def just_bob():
    start = time.time()

    bob = PPO("CnnPolicy", VectorizedClass(GetBobEnvClass(25), 6), verbose=1).learn(100000)
    evaluate(bob, 25, episodes=100)
    exit()

    done = False
    env = GetBobEnvClass(25)()
    obs = env.reset()
    while not done:
        action = bob.predict(obs)
        obs, rew, done, _ = env.step(action[0])
        env.render()

def charlie():
    start = time.time()

    bob = PPO("CnnPolicy", VectorizedClass(GetBobEnvClass(25), 6), verbose=1)
    charli = PPO("MlpPolicy", CharlieEnv(bob, 1000, 25), verbose=1).learn(100000)
    evaluate(charli.env.envs[0].bob, 25, episodes=100)
    exit()

    done = False
    env = GetBobEnvClass(25)()
    obs = env.reset()
    while not done:
        action = bob.predict(obs)
        obs, rew, done, _ = env.step(action[0])
        env.render()

def main():
    #env = BobEnv(5)
    #env.render()
   # sleep(1000)

    #just_bob()
    charlie()

if __name__ == "__main__":
    main()