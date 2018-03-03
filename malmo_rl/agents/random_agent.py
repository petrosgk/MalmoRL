from rl.agents.random import RandomAgent


class Random:
    def __init__(self, name, env):
        self.name = name
        self.nb_actions = env.available_actions
        self.agent = RandomAgent(nb_actions=self.nb_actions)
        self.agent.compile(None, None)

    def fit(self, env, nb_steps):
        self.agent.fit(env, nb_steps, action_repetition=4)

    # Fitting and testing for the random agent are the same.
    def test(self, env, nb_steps):
        self.fit(env, nb_steps)

