from stable_baselines3.common.env_util import make_vec_env


def VectorizedClass(env_class, n_envs):
    return make_vec_env(env_class, n_envs)