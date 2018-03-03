import six

import random
from mission import Mission, MissionEnvironment, MissionStateBuilder


# Simple multi-agent mission, with 2 agents and an observer based on:
# https://github.com/Microsoft/malmo/blob/master/Malmo/samples/Python_examples/multi_agent_test.py
class MultiAgent(Mission):
    def __init__(self, ms_per_tick):
        mission_name = 'multi_agent'
        agent_names = ['Agent_1', 'Agent_2', 'Observer']

        self.NUM_MOBS = 4
        self.NUM_ITEMS = 4

        mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <About>
                <Summary/>
              </About>
              <ModSettings>
                <MsPerTick>''' + str(ms_per_tick) + '''</MsPerTick>
              </ModSettings>
              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                    <StartTime>13000</StartTime>
                  </Time>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;2*4,225*22;1;" seed=""/>
                  <DrawingDecorator>
                    <DrawCuboid x1="-20" y1="200" z1="-20" x2="20" y2="200" z2="20" type="glowstone"/>
                    <DrawCuboid x1="-19" y1="200" z1="-19" x2="19" y2="227" z2="19" type="stained_glass" colour="RED"/>
                    <DrawCuboid x1="-18" y1="202" z1="-18" x2="18" y2="247" z2="18" type="air"/>
                    <DrawBlock x="0" y="226" z="0" type="fence"/>''' + self.drawMobs() + self.drawItems() + '''
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp description="" timeLimitMs="50000"/>
                </ServerHandlers>
              </ServerSection>'''

        # Add an agent section for each robot. Robots run in survival mode.
        # Give each one a wooden pickaxe for protection...
        for i in range(len(agent_names[:-1])):
            mission_xml += '''<AgentSection mode="Survival">
                <Name>''' + agent_names[i] + '''</Name>
                <AgentStart>
                  <Placement x="''' + str(random.randint(-17, 17)) + '''" y="204" z="''' + str(
                random.randint(-17, 17)) + '''"/>
                  <Inventory>
                    <InventoryObject type="iron_axe" slot="0" quantity="1"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="10"/>
                  </RewardForCollectingItem>
                  <RewardForTimeTaken initialReward="0" delta="1" density="PER_TICK"/>
                  <RewardForDamagingEntity>
                    <Mob type="Zombie" reward="1"/>
                  </RewardForDamagingEntity>
                  <RewardForSendingCommand reward="-1"/>
                  <VideoProducer>
                    <Width>512</Width>
                    <Height>512</Height>
                  </VideoProducer>
                  <ObservationFromFullStats/>
                </AgentHandlers>
              </AgentSection>'''

        # Add a section for the observer. Observer runs in creative mode.
        mission_xml += '''<AgentSection mode="Creative">
            <Name>''' + agent_names[-1] + '''</Name>
            <AgentStart>
              <Placement x="0.5" y="228" z="0.5" pitch="90"/>
            </AgentStart>
            <AgentHandlers>
              <VideoProducer>
                <Width>512</Width>
                <Height>512</Height>
              </VideoProducer>
              <ObservationFromFullStats/>
            </AgentHandlers>
          </AgentSection>'''

        mission_xml += '''</Mission>'''

        super(MultiAgent, self).__init__(mission_name=mission_name, agent_names=agent_names, mission_xml=mission_xml)

    def drawMobs(self):
        xml = ""
        for i in range(self.NUM_MOBS):
            x = str(random.randint(-17, 17))
            z = str(random.randint(-17, 17))
            xml += '<DrawEntity x="' + x + '" y="214" z="' + z + '" type="Zombie"/>'
        return xml

    def drawItems(self):
        xml = ""
        for i in range(self.NUM_ITEMS):
            x = str(random.randint(-17, 17))
            z = str(random.randint(-17, 17))
            xml += '<DrawItem x="' + x + '" y="224" z="' + z + '" type="apple"/>'
        return xml


# Define the mission environment
class MultiAgentEnvironment(MissionEnvironment):
    def __init__(self, mission_name, mission_xml, remotes, state_builder, role=0, recording_path=None,
                 force_world_reset=True):
        actions = ['move 1', 'move -1', 'turn 1', 'turn -1', 'attack 1']

        self._abs_max_reward = 10  # For reward normalization needed by some RL algorithms

        super(MultiAgentEnvironment, self).__init__(mission_name, mission_xml, actions, remotes, state_builder,
                                                    role=role, recording_path=recording_path,
                                                    force_world_reset=force_world_reset)

    # Do an action
    def step(self, action):
        action_id = action
        assert 0 <= action_id <= self.available_actions, \
            "action %d is not valid (should be in [0, %d[)" % (action_id,
                                                               self.available_actions)

        # For discrete action space, do the environment action corresponding to the action id sent by the agent
        action = self._actions[action_id]
        assert isinstance(action, six.string_types)

        if self._action_count > 0:
            if self._previous_action == 'attack 1':
                self._agent.sendCommand('attack 0')
        self._agent.sendCommand(action)
        self._previous_action = action
        self._action_count += 1

        self._await_next_obs()
        return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done, {}

    @property
    def abs_max_reward(self):
        return self._abs_max_reward


# Return low-level observations
class MultiAgentStateBuilder(MissionStateBuilder):
    def __init__(self):
        super(MultiAgentStateBuilder, self).__init__()

    def build(self, environment):
        world_obs = environment.world_observations
        return world_obs
