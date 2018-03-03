import sys
import six
import numpy as np

from mission import Mission, MissionEnvironment, MissionStateBuilder


# Simple single-agent mission
class Classroom(Mission):
    def __init__(self, ms_per_tick):
        mission_name = 'classroom'
        agent_names = ['Agent_1']

        mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <About>
                <Summary>Find the goal!</Summary>
              </About>
              <ModSettings>
                <MsPerTick>''' + str(ms_per_tick) + '''</MsPerTick>
              </ModSettings>
              <ServerSection>
                  <ServerInitialConditions>
                    <Time>
                      <StartTime>6000</StartTime>
                      <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                   </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                  <ClassroomDecorator seed="__SEED__">
                    <specification>
                      <width>7</width>
                      <height>7</height>
                      <length>15</length>
                      <pathLength>0</pathLength>
                      <divisions>
                        <southNorth>0</southNorth>
                        <eastWest>0</eastWest>
                        <aboveBelow>0</aboveBelow>
                      </divisions>
                      <horizontalObstacles>
                        <gap>1</gap>
                        <bridge>0</bridge>
                        <door>0</door>
                        <puzzle>0</puzzle>
                        <jump>0</jump>
                      </horizontalObstacles>
                      <verticalObstacles>
                        <stairs>0</stairs>
                        <ladder>0</ladder>
                        <jump>0</jump>
                      </verticalObstacles>
                      <hintLikelihood>1</hintLikelihood>
                    </specification>
                  </ClassroomDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="60000" description="out_of_time"/>
                  <ServerQuitWhenAnyAgentFinishes />
                </ServerHandlers>
              </ServerSection>
              <AgentSection mode="Survival">
                <Name>''' + agent_names[0] + '''</Name>
                <AgentStart>
                  <Placement x="-203.5" y="81.0" z="217.5"/>
                </AgentStart>
                <AgentHandlers>
                  <VideoProducer>
                    <Width>512</Width>
                    <Height>512</Height>
                  </VideoProducer>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="45">
                    <ModifierList type="deny-list">
                      <command>attack</command>
                    </ModifierList>
                  </ContinuousMovementCommands>
                  <RewardForMissionEnd rewardForDeath="-1000">
                    <Reward description="found_goal" reward="1000" />
                    <Reward description="out_of_time" reward="-1000" />
                  </RewardForMissionEnd>
                  <RewardForTouchingBlockType>
                    <Block type="gold_ore diamond_ore redstone_ore" reward="20" />
                  </RewardForTouchingBlockType>
                  <RewardForSendingCommand reward="-1"/>
                  <AgentQuitFromTouchingBlockType>
                    <Block type="gold_block diamond_block redstone_block" description="found_goal" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

        super(Classroom, self).__init__(mission_name=mission_name, agent_names=agent_names, mission_xml=mission_xml)


# Define the mission environment
class ClassroomEnvironment(MissionEnvironment):
    def __init__(self, action_space, mission_name, mission_xml, remotes, state_builder, role=0, recording_path=None):
        if action_space == 'discrete':
            actions = ['move 1', 'move -1', 'turn 1', 'turn -1']
        elif action_space == 'continuous':
            actions = ['move', 'turn']
        else:
            print('Unknown action space')
            sys.exit()

        self._abs_max_reward = 1000  # For reward normalization needed by some RL algorithms

        super(ClassroomEnvironment, self).__init__(mission_name, mission_xml, actions, remotes, state_builder,
                                                   role=role, recording_path=recording_path)

    # Do an action. Supports either a single discrete action or a list/tuple of continuous actions
    def step(self, action):
        if isinstance(action, list) or isinstance(action, tuple):
            assert 0 <= len(action) <= self.available_actions, \
                "action list is not valid (should be of length [0, %d[)" % (
                    self.available_actions)
        else:
            action_id = action
            assert 0 <= action_id <= self.available_actions, \
                "action %d is not valid (should be in [0, %d[)" % (action_id,
                                                                   self.available_actions)

        if isinstance(action, list) or isinstance(action, tuple):
            # For continuous action space, do the environment action(s) by the amount sent by the agent for each
            action = [self._actions[i] + ' ' + str(action[i]) for i in range(len(action))]
        else:
            # For discrete action space, do the environment action corresponding to the action id sent by the agent
            action = self._actions[action_id]
            assert isinstance(action, six.string_types)

        if isinstance(action, list):
            for i in range(len(action)):
                self._agent.sendCommand(action[i])
        else:
            self._agent.sendCommand(action)
        self._previous_action = action
        self._action_count += 1

        self._await_next_obs()
        return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done, {}

    @property
    def abs_max_reward(self):
        return self._abs_max_reward


# Define a mission state builder
class ClassroomStateBuilder(MissionStateBuilder):
    """
    Generate RGB frame state resizing to the specified width/height and depth
    """
    def __init__(self, width, height, grayscale):
        assert width > 0, 'width should be > 0'
        assert height > 0, 'height should be > 0'

        self._width = width
        self._height = height
        self._gray = bool(grayscale)

        super(ClassroomStateBuilder, self).__init__()

    def build(self, environment):

        img = environment.frame

        if img is not None:
            img = img.resize((self._width, self._height))

            if self._gray:
                img = img.convert('L')
            return np.array(img)
        else:
            return np.zeros((self._width, self._height, 1 if self._gray else 3)).squeeze()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def grayscale(self):
        return self._gray
