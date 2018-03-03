import os
from malmopy.environment.malmo import MalmoEnvironment, MalmoStateBuilder


class Mission(object):
    def __init__(self, mission_name, agent_names, mission_xml):
        assert mission_name is not None, 'Mission must have a name'
        assert agent_names is not None and len(agent_names) > 0, 'Mission must have at least 1 agent'
        assert mission_xml is not None, 'A mission XML must be defined'

        self.mission_name = mission_name
        self.agent_names = agent_names
        self.mission_xml = mission_xml


class MissionEnvironment(MalmoEnvironment):
    def __init__(self, mission_name, mission_xml, actions, remotes, state_builder, role=0, turn_based=False,
                 force_world_reset=False, recording_path=None):
        assert state_builder is not None, 'A mission state builder must be defined'

        self._mission_name = mission_name
        self._action_space = [action_id for action_id, action in enumerate(actions)]

        super(MissionEnvironment, self).__init__(mission_xml, actions, remotes, role=role, turn_based=turn_based,
                                                 recording_path=recording_path, force_world_reset=force_world_reset)

        self._user_defined_builder = state_builder

    # Do an action in the environment
    def step(self, action):
        super(MissionEnvironment, self).do(action)

    @property
    def state(self):
        return self._user_defined_builder.build(self)

    @property
    def mission_name(self):
        return self._mission_name

    @property
    def action_space(self):
        return self._action_space


class MissionStateBuilder(MalmoStateBuilder):
    def __init__(self):
        super(MissionStateBuilder, self).__init__()

    def build(self, environment):
        super(MissionStateBuilder, self).build(environment)
