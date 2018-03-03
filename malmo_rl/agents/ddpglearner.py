import os
import numpy as np
from agent import BaseAgent
from keras.optimizers import Adam
import keras.backend as K
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.random import GaussianWhiteNoiseProcess
from rl.agents.ddpg import DDPGAgent
from malmo_rl.model import Minecraft_DDPG, Minecraft_DDPG_LSTM


class DDPGLearner(BaseAgent):
    def __init__(self, name, env, grayscale, width, height):
        super(DDPGLearner, self).__init__(name=name, env=env)

        self.nb_actions = self.env.available_actions
        self.abs_max_reward = self.env.abs_max_reward
        self.mission_name = self.env.mission_name

        self.grayscale = grayscale
        self.width = width
        self.height = height

        self.recurrent = False  # Use LSTM
        self.batch_size = 32
        self.window_length = 4

        if not self.recurrent:
            self.actor, self.critic, self.action_input = Minecraft_DDPG(self.window_length, self.grayscale, self.width,
                                                                        self.height, self.nb_actions)
        else:
            self.actor, self.critic, self.action_input = Minecraft_DDPG_LSTM(self.window_length, self.grayscale,
                                                                             self.width, self.height, self.nb_actions)

        # Replay memory
        self.memory = SequentialMemory(limit=1000000, window_length=self.window_length)

        # Add random noise for exploration
        self.random_process = GaussianWhiteNoiseProcess(mu=0.0, sigma=1.0, size=self.nb_actions)

        '''
        # We can also generate exploration noise with different parameters for each action. This is because we may want
        # eg. the agent to be more likely to explore moving forward than backward. In that case, a list or tuple of
        # random processes, one for each action, must be passed to the agent.
        # For example:

        self.random_process = []
        self.random_process.append(GaussianWhiteNoiseProcess(mu=1.5, sigma=1.0))  # For moving
        self.random_process.append(GaussianWhiteNoiseProcess(mu=0.0, sigma=1.0))  # For turning
        '''

        self.processor = MalmoProcessor(self.grayscale, self.window_length, self.recurrent, self.abs_max_reward)
        self.agent = DDPGAgent(actor=self.actor, critic=self.critic, critic_action_input=self.action_input,
                               nb_actions=self.nb_actions, memory=self.memory, batch_size=self.batch_size,
                               processor=self.processor, random_process=self.random_process, gamma=0.99,
                               nb_steps_warmup_actor=10000, nb_steps_warmup_critic=10000, target_model_update=1e-3)
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    def fit(self, env, nb_steps):
        weights_dir = 'weights/{}'.format(self.mission_name)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        weights_path = os.path.join(weights_dir, '{}'.format(self.name))
        callbacks = [ModelIntervalCheckpoint(weights_path, interval=10000, verbose=1)]
        self.agent.fit(env, nb_steps, action_repetition=4, callbacks=callbacks, verbose=1, log_interval=10000,
                       test_interval=10000, test_nb_episodes=10, test_action_repetition=4, test_visualize=False)

    def test(self, env, nb_episodes):
        self.agent.test(env, nb_episodes, action_repetition=4, callbacks=None, verbose=1, visualize=False)

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
                if K.image_data_format() == 'channels_last':
                    processed_batch = np.concatenate(states, axis=3)
                else:
                    processed_batch = np.concatenate(states, axis=3).transpose((0, 3, 1, 2))  # Channels-first order
            else:
                if K.image_data_format() == 'channels_last':
                    processed_batch = batch
                else:
                    processed_batch = batch.transpose((0, 1, 4, 2, 3))  # Channels-first order
        else:
            if not self.recurrent:
                if K.image_data_format() == 'channels_last':
                    processed_batch = batch.transpose((0, 2, 3, 1))
                else:
                    processed_batch = batch
            else:
                if K.image_data_format() == 'channels_last':
                    processed_batch = np.expand_dims(batch, axis=4)
                else:
                    processed_batch = np.expand_dims(batch, axis=2)
        processed_batch = processed_batch.astype('float32') / 255.

        return processed_batch

    def process_action(self, action):
        action = list(action)
        return action

    def process_reward(self, reward):
        if self.abs_max_reward:
            return reward / self.abs_max_reward
        else:
            return reward
