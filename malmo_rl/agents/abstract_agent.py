from agent import BaseAgent, Observer

from malmo_rl.agents.random_agent import Random
from malmo_rl.agents.qlearner import QLearner
from malmo_rl.agents.ddpglearner import DDPGLearner


class AbstractAgent(BaseAgent):
    def __init__(self, name, env, agent_type, **kwargs):
        if agent_type == 'random':
            self.agent = Random(name, env)
        elif agent_type == 'dqn':
            self.agent = QLearner(name, env, kwargs['grayscale'], kwargs['width'], kwargs['height'])
        elif agent_type == 'ddpg':
            self.agent = DDPGLearner(name, env, kwargs['grayscale'], kwargs['width'], kwargs['height'])
        elif agent_type == 'observer':
            self.agent = Observer(name, env)
        else:
            RuntimeError('Unknown agent type')
        super(AbstractAgent, self).__init__(name, env)

    def fit(self, env, nb_steps):
        self.agent.fit(env, nb_steps)

    def test(self, env, nb_episodes):
        return self.agent.test(env, nb_episodes)

    def save(self, out_dir):
        self.agent.save(out_dir)

    def load(self, out_dir):
        self.agent.load(out_dir)