import gym as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from bobenv import GetBobEnvClass
from evaluatebob import evaluate
from vectorizeenv import VectorizedClass
import numpy as np

class CharlieEnv(gym.Env):
    def __init__(self, bob, t, maxsize):
        self.bob = bob
        self.t = t
        #self.obs_factor = 4
        self.observation_space = gym.spaces.Box(0,1, shape=(10,), dtype=float) # scale rew/size
        self.action_space = gym.spaces.Discrete(maxsize-4)

        self.counter = 0

    def reset(self):
        return np.full((10,), 0.5)

    def step(self, action: int):
        """
        print("c:", self.counter)
        self.counter += 1
        return np.random.randint(0, 1, (self.t,)), 0, False, {}"""

        action += 4
        env = VectorizedClass(GetBobEnvClass(action), 6)
        self.bob.set_env(env)
        #a = evaluate(self.bob, action, 5)

        obssss = []
        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                pass
            def _on_rollout_end(self):
                x = self.model.rollout_buffer.rewards.tolist()
                obssss.extend(x)

        self.bob = self.bob.learn(self.t, callback=Callback())#, callback=GetReward())

        buffer = self.bob.rollout_buffer


        fuck = np.array(obssss).mean(axis=1)

        """
        size = fuck.size
        fuck15 = fuck.reshape((size, size//self.t))
        fuck2 = fuck15.mean(axis=0)
        fuck3 = np.zeros((200,)) if fuck2.max() == 0 else fuck2 / fuck2.max()
        """

        tempargh = chunkmean(fuck, 10)

        bminusa = tempargh[-1] - tempargh[0]# obssss[-1].mean()-obssss[0].mean()
        #print("c:",self.counter)
        #self.counter += 1
        print("cstep")

       # b = evaluate(self.bob, action, 5, verbose=True)
        return np.zeros((10,)) if tempargh.max() == 0 else tempargh / tempargh.max(), bminusa, False, {}

def chunkmean(arr, nb_chunks):
    return arr[:arr.size//nb_chunks*nb_chunks].reshape((nb_chunks, arr.size//nb_chunks)).mean(axis=1)

def GetCharlieEnvClass(bob, t, maxsize):
    def temp():
        return CharlieEnv(bob, t, maxsize)
    return temp
