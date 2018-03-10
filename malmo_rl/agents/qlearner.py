import os
import numpy as np
from agent import BaseAgent

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras.backend import tensorflow_backend, image_data_format
from keras.optimizers import Adam

from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy
from rl.policy import BiasedEpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from malmo_rl.model import Minecraft, Minecraft_LSTM


class QLearner(BaseAgent):
    def __init__(self, name, env, grayscale, width, height):
        super(QLearner, self).__init__(name=name, env=env)

        self.nb_actions = env.available_actions
        self.abs_max_reward = env.abs_max_reward
        self.mission_name = env.mission_name

        self.grayscale = grayscale
        self.width = width
        self.height = height

        self.recurrent = False  # Use LSTM
        self.batch_size = 32
        self.window_length = 4

        if tf:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            tensorflow_backend.set_session(session=sess)

        if not self.recurrent:
            self.model = Minecraft(self.window_length, self.grayscale, self.width, self.height, self.nb_actions)
        else:
            self.model = Minecraft_LSTM(self.window_length, self.grayscale, self.width, self.height, self.nb_actions)

        # Replay memory
        self.memory = SequentialMemory(limit=1000000, window_length=self.window_length)

        '''
        Select a policy. We use eps-greedy action selection, which means that a random action is selected
        with probability eps. We can specify a custom biased probability distribution p for selecting random action,
        so that the agent is more likely to choose some actions when exploring over others. For example,
        if the possible actions are [move forward, move backward, turn right, turn left] and p = [0.6, 0.0, 0.2, 0.2] 
        the agent will go 60% forward, 0% backward, 20% left and 20% right when exploring.
        If p == None, the default uniform distribution is used.
        '''
        self.policy = LinearAnnealedPolicy(BiasedEpsGreedyQPolicy(nb_actions=self.nb_actions, p=None),
                                           attr='eps', value_max=1., value_min=.05, value_test=.005, nb_steps=1000000)

        self.processor = MalmoProcessor(self.grayscale, self.window_length, self.recurrent, self.abs_max_reward)
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, policy=self.policy, test_policy=self.policy,
                              memory=self.memory, batch_size=self.batch_size, processor=self.processor,
                              nb_steps_warmup=50000, gamma=.99, target_model_update=10000, enable_double_dqn=True,
                              enable_dueling_network=True)
        self.agent.compile(Adam(lr=.00025), metrics=['mae'])

    def fit(self, env, nb_steps):
        weights_dir = 'weights/{}'.format(self.mission_name)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        weights_path = os.path.join(weights_dir, '{}'.format(self.name))
        callbacks = [ModelIntervalCheckpoint(weights_path, interval=10000, verbose=1)]
        self.agent.fit(env, nb_steps, action_repetition=4, callbacks=callbacks, verbose=1, log_interval=10000,
                       test_interval=10000, test_nb_episodes=10, test_action_repetition=4, test_visualize=False)

    def test(self, env, nb_episodes):
        self.agent.test(env, nb_episodes, action_repetition=4, verbose=1, visualize=False)

    def save(self, out_dir):
        self.agent.save_weights(out_dir, overwrite=True)

    def load(self, out_dir):
        self.agent.load_weights(out_dir)


class MalmoProcessor(Processor):
    def __init__(self, grayscale, window_length, recurrent, abs_max_reward):
        self.grayscale = grayscale
        self.window_length = window_length
        self.recurrent = recurrent
        self.abs_max_reward = abs_max_reward

    def process_state_batch(self, batch):
        if not self.grayscale:
            if not self.recurrent:
                states = []
                # Get each state in the batch
                for i in range(self.window_length):
                    states.append(batch[:, i, :, :, :])
                # Concatenate states in the batch along the channel axis
                if image_data_format() == 'channels_last':
                    processed_batch = np.concatenate(states, axis=3)
                else:
                    processed_batch = np.concatenate(states, axis=3).transpose((0, 3, 1, 2))  # Channels-first order
            else:
                if image_data_format() == 'channels_last':
                    processed_batch = batch
                else:
                    processed_batch = batch.transpose((0, 1, 4, 2, 3))  # Channels-first order
        else:
            if not self.recurrent:
                if image_data_format() == 'channels_last':
                    processed_batch = batch.transpose((0, 2, 3, 1))
                else:
                    processed_batch = batch
            else:
                if image_data_format() == 'channels_last':
                    processed_batch = np.expand_dims(batch, axis=4)
                else:
                    processed_batch = np.expand_dims(batch, axis=2)
        processed_batch = processed_batch.astype('float32') / 255.

        return processed_batch

    def process_reward(self, reward):
        if self.abs_max_reward:
            return reward / self.abs_max_reward
        else:
            return reward
