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
        self.obs_factor = 4
        self.observation_space = gym.spaces.Box(0,1, shape=(t//self.obs_factor,), dtype=float) # scale rew/size
        self.action_space = gym.spaces.Discrete(maxsize-4)

    def reset(self):
        return np.random.randint(0,1,(self.t//self.obs_factor,))

    def step(self, action: int):
        action += 4
        env = VectorizedClass(GetBobEnvClass(action), 6)
        self.bob.set_env(env)
        a = evaluate(self.bob, action, 5)

        obssss = []
        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                pass
            def _on_rollout_end(self):
                x = self.model.rollout_buffer.rewards.tolist()[0]
                obssss.extend(x)

        self.bob = self.bob.learn(self.t, callback=Callback())#, callback=GetReward())

        buffer = self.bob.rollout_buffer

        obssss = np.array(obssss[:(self.t//self.obs_factor*self.obs_factor)])
        fuck = obssss
        fuck15 = fuck.reshape((self.obs_factor, self.t//self.obs_factor))
        fuck2 = fuck15.mean(axis=0)

        b = evaluate(self.bob, action, 5, verbose=True)
        return fuck2, b-a, False, {}

def GetCharlieEnvClass(bob, t, maxsize):
    def temp():
        return CharlieEnv(bob, t, maxsize)
    return temp
