# initial-trial
Tests on minigrid

Basically the env is an empty room with one agent and one goal. Reward is super sparse,
only output when bob reaches goal.
        
    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

Then, I'm just scaling it by the env size (makes sense, bigger rooms == harder task).

Also the goal is in a random position in the env.


# Results:

## Sanity check:

Env: 

    a++
    +g+
    +++

Using a size 5 env, 3000 eval runs yields 2.9835 mean reward.
Training a PPO bob for 98k ep and evaling on the same yields 4.7862.

## Real task:

Let's say that we actually want bob to perform well on size 25.

Untrained Bob: 0.23. Only evaled on 100 episodes because evaling on something that sparse
is a bitch.

Trained for 100k: 0.776