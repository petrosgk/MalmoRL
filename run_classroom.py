import os
from argparse import ArgumentParser
from multiprocessing import Process, active_children
from time import sleep
from common import parse_clients_args

from missions.classroom import Classroom


def agent_factory(name, role, clients, agent_type, steps, mission, action_space, mode):
    from missions.classroom import ClassroomEnvironment, ClassroomStateBuilder

    from malmo_rl.agents.abstract_agent import AbstractAgent

    clients = parse_clients_args(clients)

    recording_dir = 'records/{}'.format(mission.mission_name)
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
    recording_path = os.path.join(recording_dir, '{}.tgz'.format(name))

    state_builder = ClassroomStateBuilder(width=32, height=32, grayscale=True)
    env = ClassroomEnvironment(action_space, mission.mission_name, mission.mission_xml, clients, state_builder,
                               role=role, recording_path=recording_path)

    if 'Observer' in name:
        agent_type = 'observer'

    agent = AbstractAgent(name, env, agent_type, grayscale=state_builder.grayscale, width=state_builder.width,
                          height=state_builder.height)
    print(name + ' initialized.')

    weights_filename = 'weights/{}/{}_{}'.format(mission.mission_name, agent_type, name)
    if mode == 'training':
        agent.fit(env, steps)
        agent.save(weights_filename)
    else:
        agent.load(weights_filename)
        agent.test(env, nb_episodes=10)


def run_experiment(agents_def):
    assert len(agents_def) >= 1, 'Not enough agents (required: >= 1, got: %d)' \
                                 % len(agents_def)

    for agent in agents_def:
        p = Process(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

    try:
        # wait until all agents are finished
        while len(active_children()) > 0:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')


if __name__ == '__main__':
    arg_parser = ArgumentParser('Malmo experiment')
    arg_parser.add_argument('--ms-per-tick', type=int, default=50,
                            help='Malmo running speed')
    arg_parser.add_argument('--clients', default='clients.txt',
                            help='.txt file with client(s) IP addresses')
    arg_parser.add_argument('--steps', type=int, default=1000000,
                            help='Number of steps to train for')
    arg_parser.add_argument('--action-space', default='discrete',
                            help='Action space to use (discrete, continuous)')
    arg_parser.add_argument('--agents', default=['random'], nargs='+',
                            help='Agent(s) to use (default is 1 Random agent)')
    arg_parser.add_argument('--mode', default='training',
                            help='Training or testing mode')
    args = arg_parser.parse_args()

    ms_per_tick = args.ms_per_tick
    clients = args.clients
    steps = args.steps
    action_space = args.action_space
    agents = args.agents
    mode = args.mode

    mission = Classroom(ms_per_tick)
    mission_agent_names = mission.agent_names

    assert len(agents) == len(mission_agent_names), '1 agent must be specified for each mission agent name'

    clients = open(clients, 'r').read().splitlines()
    print('Clients: {}'.format(clients))
    assert len(clients) >= len(mission_agent_names), '1 Malmo client for each agent must be specified in clients.txt'

    # Setup agents
    agents_def = [{'name': agent_name, 'role': idx, 'clients': clients, 'agent_type': agents[idx], 'steps': steps,
                   'mission': mission, 'action_space': action_space, 'mode': mode}
                  for idx, agent_name in enumerate(mission_agent_names)]

    run_experiment(agents_def)
