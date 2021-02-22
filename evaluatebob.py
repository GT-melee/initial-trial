from tqdm import trange

from bobenv import BobEnv


def evaluate(bob, size, episodes=3000, verbose=False):
    nb_steps_each = []
    for i in range(episodes):
        env = BobEnv(size)
        done = False
        obs = env.reset()
        rews = 0
        while not done:
            obs, rew, done, _ = env.step(bob.predict(obs)[0])
            rews += rew
        nb_steps_each.append(rews)
    if verbose:
        print(sum(nb_steps_each)/3000)
    return sum(nb_steps_each)/3000