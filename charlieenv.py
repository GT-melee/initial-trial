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
        self.observation_space = gym.spaces.Box(0,8, shape=(t,), dtype=int) # scale rew/size
        self.action_space = gym.spaces.Discrete(maxsize-3)

    def reset(self):
        return np.random.randint(0,8,(1000,))

    def step(self, action: int):
        class GetReward(BaseCallback):
            def __init__(self):
                super(GetReward, self).__init__()

            def _on_step(self) -> bool:
                # Convert np.bool to bool, otherwise callback() is False won't work
                continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
                if self.verbose > 0 and not continue_training:
                    print(
                        f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                        f" is above the threshold {self.reward_threshold}"
                    )
                return continue_training


        env = VectorizedClass(GetBobEnvClass(action+3), 6)
        self.bob.set_env(env)
        a = evaluate(self.bob, action, 10)
        self.bob = self.bob.learn(self.t)
        b = evaluate(self.bob, action, 10)
        return np.random.randint(0,8,(1000,)), b-a, False, {}

def GetCharlieEnvClass(bob, t, maxsize):
    def temp():
        return CharlieEnv(bob, t, maxsize)
    return temp
