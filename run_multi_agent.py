import os
from argparse import ArgumentParser
from multiprocessing import Process, active_children
from time import sleep

from missions.multi_agent import MultiAgent


def parse_clients_args(args_clients):
    """
    Return an array of tuples (ip, port) extracted from ip:port string
    :param args_clients:
    :return:
    """
    return [str.split(str(client), ':') for client in args_clients]


def agent_factory(name, role, clients, agent_type, steps, mission, mode):
    from missions.multi_agent import MultiAgentEnvironment, MultiAgentStateBuilder

    from malmo_rl.agents.abstract_agent import AbstractAgent

    clients = parse_clients_args(clients)

    recording_dir = 'records/{}'.format(mission.mission_name)
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
    recording_path = os.path.join(recording_dir, '{}.tgz'.format(name))

    state_builder = MultiAgentStateBuilder()
    env = MultiAgentEnvironment(mission.mission_name, mission.mission_xml, clients, state_builder,
                                role=role, recording_path=recording_path)

    if 'Observer' in name:
        agent_type = 'observer'

    agent = AbstractAgent(name, env, agent_type)
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
    arg_parser.add_argument('--agents', default='random random observer', nargs='+',
                            help='Agent(s) to use (default is 2 Random agents and an Observer)')
    arg_parser.add_argument('--mode', default='training',
                            help='Training or testing mode')
    args = arg_parser.parse_args()

    ms_per_tick = args.ms_per_tick
    clients = args.clients
    steps = args.steps
    agents = args.agents
    mode = args.mode

    mission = MultiAgent(ms_per_tick)
    mission_agent_names = mission.agent_names

    assert len(agents) == len(mission_agent_names), '1 agent must be specified for each mission agent name'

    clients = open(clients, 'r').read().splitlines()
    print('Clients: {}'.format(clients))
    assert len(clients) >= len(mission_agent_names), '1 Malmo client for each agent must be specified in clients.txt'

    # Setup agents
    agents_def = [{'name': agent_name, 'role': idx, 'clients': clients, 'agent_type': agents[idx], 'steps': steps,
                   'mission': mission, 'mode': mode}
                  for idx, agent_name in enumerate(mission_agent_names)]

    run_experiment(agents_def)
