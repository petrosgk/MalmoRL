# MalmoRL
A framework for training Reinforcement Learning agents in Minecraft with [Project Malmö](https://github.com/Microsoft/malmo). I've built it for my own research and I hope it's useful to others as well. It is partially based on code provided with the [The Malmo Collaborative AI Challenge](https://github.com/Microsoft/malmo-challenge), extended to support more Malmö mission environments. It should also be easy to extend further to support the needs of more environments.   

![DRQN with biased ε-greedy](pools_dqn.gif)
![DRQN with biased ε-greedy](rooms_dqn.gif)
![DRQN with biased ε-greedy](obstacles_dqn.gif)
![DRQN with biased ε-greedy](labyrinth_dqn.gif)

Work in progress...

### Define a mission
Create `missions/<your_mission>.py`. Inside it you must define 3 classses: 
1. `Mission`, where you should define at least the `mission_name` the `agent_names` and the `mission_xml` description.

2. `MissionEnvironment`, where you should define at least the available `actions` in the environment. You can optionally define several other aspects of the environment, like how you want actions sent by the agent to be handled etc. by overriding the respective methods.

3. `MissionStateBuilder`, where you can define the states (frames, observations etc.) produced by the environment. You must override the `build()` method to create and return states to the agent.

Take a look at the included `missions/classroom.py` and `missions/multi_agent.py` for more concrete examples.
