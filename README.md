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


100%|██████████| 100/100 [01:36<00:00,  1.03it/s]
0.6008936666666664
For 100000 we took 179.279301404953 and got 0.6008936666666664
100%|██████████| 100/100 [00:17<00:00,  5.78it/s]
0.7954943333333334
For 500000 we took 824.559549331665 and got 0.7954943333333334
100%|██████████| 100/100 [00:09<00:00, 10.87it/s]
0.8126093333333336
For 1000000 we took 1658.4472596645355 and got 0.8126093333333336
100%|██████████| 100/100 [00:19<00:00,  5.14it/s]
0.7899263333333338
For 5000000 we took 8004.223884344101 and got 0.7899263333333338

For 16666 we took 39424.272664785385 and got 0.22252
For 83333 we took 184250.70336389542 and got 0.21111083333333333