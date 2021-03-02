import time

import matplotlib

from charlieenv import CharlieEnv
from evaluatebob import evaluate
from vectorizeenv import VectorizedClass

matplotlib.use('TkAgg')
from stable_baselines3 import PPO
from bobenv import GetBobEnvClass


def just_bob():
    for i in [100000, 500000, 1000000, 5000000]:
        start = time.time()
        bob = PPO("CnnPolicy", VectorizedClass(GetBobEnvClass(25), 6), verbose=0).learn(i)
        end = time.time()
        print(f"For {i} we took {end-start} and got {evaluate(bob, 25, episodes=100)}")
    exit()

    done = False
    env = GetBobEnvClass(25)()
    obs = env.reset()
    while not done:
        action = bob.predict(obs)
        obs, rew, done, _ = env.step(action[0])
        env.render()

def charlie():
    for i in [100000//6, 500000//6, 1000000//6, 5000000//6]:
        start = time.time()
        bob = PPO("CnnPolicy", VectorizedClass(GetBobEnvClass(10), 6), verbose=0, n_steps=200)
        charli = PPO("MlpPolicy", CharlieEnv(bob, t=200, maxsize=10), verbose=0, n_steps=1000).learn(i)
        end = time.time()
        print(f"For {i} we took {end-start} and got {evaluate(bob, 10, episodes=100)}")
    exit()

    done = False
    env = GetBobEnvClass(25)()
    obs = env.reset()
    while not done:
        action = bob.predict(obs)
        obs, rew, done, _ = env.step(action[0])
        env.render()

def main():
    """
    start = time.time()

    bob = PPO("CnnPolicy", VectorizedClass(GetBobEnvClass(25), 6), verbose=1)#.learn(100000)
    #evaluate(bob, 25, episodes=100)


    done = False
    env = GetBobEnvClass(25)()
    obs = env.reset()
    while not done:
        action = bob.predict(obs)
        obs, rew, done, _ = env.step(action[0])
        env.render()
    #env = BobEnv(5)
    #env.render()
   # sleep(1000)

    #just_bob()"""
    charlie()

if __name__ == "__main__":
    #main()
    charlie()