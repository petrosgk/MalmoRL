from agent import BaseAgent
from rl.agents.random import RandomAgent


class Random(BaseAgent):
    def __init__(self, name, env):
        super(Random, self).__init__(name, env)

        self.nb_actions = env.available_actions
        self.agent = RandomAgent(nb_actions=self.nb_actions)

    def fit(self, env, nb_steps):
        # Crashes for verbose=1
        self.agent.fit(env, nb_steps, action_repetition=4)

    # Fitting and testing for the random agent are the same.
    def test(self, env, nb_steps):
        self.fit(env, nb_steps)
