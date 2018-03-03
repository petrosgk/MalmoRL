class BaseAgent(object):
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def fit(self, env, nb_steps):
        raise NotImplementedError

    def test(self, env, nb_episodes):
        raise NotImplementedError

    def save(self, out_dir):
        raise NotImplementedError

    def load(self, out_dir):
        raise NotImplementedError


class Observer(BaseAgent):
    def __init__(self, name, env):
        super(Observer, self).__init__(name, env.available_actions)

    def fit(self, env, nb_steps):
        env.reset()

        for step in range(1, nb_steps + 1):

            # Check if env needs reset
            if env.done:
                env.reset()

            # Select an action to advance the environment one step
            # The action will be ignored by Malmo
            action = 0  # Just return the 1st action
            env.do(action)

    def test(self, env, nb_steps):
        # Fitting and testing for the observer agent are the same.
        self.fit(env, nb_steps)

    def save(self, out_dir):
        pass

    def load(self, out_dir):
        pass
