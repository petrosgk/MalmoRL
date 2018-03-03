from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import LSTM, TimeDistributed
from keras.layers import concatenate
from keras.layers import RepeatVector
from keras.initializers import RandomUniform
import keras.backend as K


def Minecraft(window_length, grayscale, width, height, nb_actions):
    assert width == 32 and height == 32, \
        'Model accepts 32x32 input size, got {}x{}'.format(width, height)
    if grayscale:
        channels = 1
    else:
        channels = 3
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, window_length * channels)
    else:
        input_shape = (window_length * channels, 32, 32)

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())

    return model


def Minecraft_LSTM(window_length, grayscale, width, height, nb_actions):
    assert width == 32 and height == 32, \
        'Model accepts 32x32 input size, got {}x{}'.format(width, height)
    if grayscale:
        channels = 1
    else:
        channels = 3
    if K.image_data_format() == 'channels_last':
        input_shape = (window_length, 32, 32, channels)
    else:
        input_shape = (window_length, channels, 32, 32)

    inputs = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))(inputs)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, stateful=False, return_sequences=False)(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())

    return model


def Atari2015(window_length, grayscale, width, height, nb_actions):
    # We use the same model that was described by Mnih et al. (2015).
    assert width == 84 and height == 84, 'Model accepts 84x84 input size'
    if grayscale:
        channels = 1
    else:
        channels = 3
    if K.image_data_format() == 'channels_last':
        input_shape = (84, 84, window_length * channels)
    else:
        input_shape = (window_length * channels, 84, 84)

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())

    return model


def Atari2015_LSTM(window_length, grayscale, width, height, nb_actions):
    # We use the same model that was described by Mnih et al. (2015) with fully connected layer replaced
    # by an LSTM, as in https://arxiv.org/abs/1507.06527
    assert width == 84 and height == 84, 'Model accepts 84x84 input size'
    if grayscale:
        channels = 1
    else:
        channels = 3
    input_shape = (window_length, channels, 84, 84)

    inputs = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(inputs)
    x = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(512, stateful=False, return_sequences=False)(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())

    return model


def Minecraft_DDPG(window_length, grayscale, width, height, nb_actions):
    assert width == 32 and height == 32, 'Model accepts 32x32 input size'
    if grayscale:
        channels = 1
    else:
        channels = 3
    if K.image_data_format() == 'channels_last':
        observation_shape = (32, 32, window_length * channels)
    else:
        observation_shape = (window_length * channels, 32, 32)

    # Build actor and critic networks
    inputs = Input(shape=observation_shape)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(nb_actions, activation='tanh', kernel_initializer=RandomUniform(-3e-4, 3e-4))(x)
    actor = Model(inputs=inputs, outputs=x)
    print(actor.summary())

    # critic network has 2 inputs, one action input and one observation input.
    action_input = Input(shape=(nb_actions, ), name='action_input')
    observation_input = Input(shape=observation_shape, name='observation_input')
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(observation_input)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = concatenate([x, action_input])  # Actions are not included until the 2nd dense layer
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='linear', kernel_initializer=RandomUniform(-3e-4, 3e-4))(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    return actor, critic, action_input


def Minecraft_DDPG_LSTM(window_length, grayscale, width, height, nb_actions):
    assert width == 32 and height == 32, 'Model accepts 32x32 input size'
    if grayscale:
        channels = 1
    else:
        channels = 3
    if K.image_data_format() == 'channels_last':
        observation_shape = (window_length, 32, 32, channels)
    else:
        observation_shape = (window_length, channels, 32, 32)

    # Build actor and critic networks
    inputs = Input(shape=observation_shape)
    x = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))(inputs)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, stateful=False, return_sequences=True)(x)
    x = LSTM(256, stateful=False, return_sequences=False)(x)
    x = Dense(nb_actions, activation='tanh', kernel_initializer=RandomUniform(-3e-4, 3e-4))(x)
    actor = Model(inputs=inputs, outputs=x)
    print(actor.summary())

    # critic network has 2 inputs, one action input and one observation input.
    action_input = Input(shape=(nb_actions, ), name='action_input')
    observation_input = Input(shape=observation_shape, name='observation_input')
    x = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))(observation_input)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, stateful=False, return_sequences=True)(x)
    repeated_action_input = RepeatVector(window_length)(action_input)
    x = concatenate([x, repeated_action_input])  # Actions are not included until the 2nd dense layer
    x = LSTM(256, stateful=False, return_sequences=False)(x)
    x = Dense(1, activation='linear', kernel_initializer=RandomUniform(-3e-4, 3e-4))(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    return actor, critic, action_input