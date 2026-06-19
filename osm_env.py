from enum import IntEnum
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import osmnx as ox


register( # for registering the environment with Gymnasium
    id="osm_map",
    entry_point="osm_env:OSMMap"
)


class AgentAction(IntEnum):
    """Action spce of the agent"""

    ADD_NODE = 0        # add new node in loop
    REMOVE_NODE = 1     # remove node in loop
    SWAP_NODE = 2       # replace a node in the loop
    ACCEPT = 3

class OSMMap(gym.Env):
    """Open Street Map environment for closed walkable loops."""
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, starting_point="Wemmel, Belgium", duration=None, render_mode=None, max_steps=50):
        
        """Initialize the obs space, action space, and map"""
        
        lat, lon = 50.9079, 4.3005                                      # coordinates of Wemmel Belgium
        G = ox.graph_from_place(starting_point, network_type="walk")    # graph loading with the starting node
        G = ox.truncate.largest_component(G, strongly=True)             # to keep only the connected part of the graph
        self.starting_node = ox.distance.nearest_nodes(G, lon, lat)     # get the nearest graph node to the starting node
        G = ox.project_graph(G)                                         # project the graph into a meter based system
        
        self.G = G
        self.nodes = list(self.G.nodes)
        self.walking_speed = 5000 / 60                                 # walking speed is fixed at 5km/h
        
        self.duration = duration                                       # None = random, or fixed value (5-50)
        self.max_steps = max_steps                                     # maximum 50 steps per episode
        self.min_node = 1                                              # minimum number of nodes in a closed walkable path
        self.max_node = 10                                             # maximum number of nodes in a closed walkable path
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        
        # Observation: 
        # target duration,
        # actual duration, 
        # difference in duration, 
        # number of nodes in loop, 
        # episode ratio (completed steps /total steps),
        # overlap ratio (ratio of repeated nodes in a loop).
        self.observation_space = spaces.Box(
            low=np.array( [5,  0,  0, self.min_node, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 50, self.max_node, 1, 1], dtype=np.float32),
        )
    
    def reset(self, seed=None, options=None):
        """Reset method for the environment"""
        super().reset(seed=seed)
        if self.duration is not None:
            self.target_duration = float(self.duration)
        else:
            self.target_duration = float(self.np_random.uniform(5, 50))
        self.current_loop = self._generate_random_loop()                        # generate a random walkable path
        self.actual_duration = self._compute_duration(self.current_loop)        # compute the actual duration
        self.current_step = 0                                                   # set the initial step as 0
        obs = self._get_observation()                                           # get the observation space values
        info = {}
        return obs, info

    def step(self, action):
        """Step method for the environment"""
        action = AgentAction(int(action))
        self.current_step += 1
        truncated = False
        terminated = False
        old_diff = abs(self.target_duration - self.actual_duration)
        
        if action == AgentAction.ADD_NODE:                                     # when the agent selects the ADD_NODE action
            self._add_node()                                                   # add a new node in the loop
        elif action == AgentAction.REMOVE_NODE:                                # when the agent selects the REMOVE_NODE action 
            self._remove_node()                                                # remove a node in the loop
        elif action == AgentAction.SWAP_NODE:                                  # when the agent selects the SWAP_NODE action
            self._replace_node()                                               # replace a node in the loop
        elif action == AgentAction.ACCEPT:                                     # when the agent selects the ACCEPT action
            quality = self._quality_score()                                    # compute the quality of the loop and terminate the episode
            reward = quality
            terminated = True   
            obs = self._get_observation()
            info = {}
            return obs, reward, terminated, truncated, info

        self.actual_duration = self._compute_duration(self.current_loop)        # compute the actual duration
        new_diff = abs(self.target_duration - self.actual_duration)             # compute the absolute difference
        reward = (old_diff - new_diff) / self.target_duration                   # calculate the reward
        reward -= 0.01                                                          # small penlty to prevent editing for every step

        if self.current_step >= self.max_steps:                                 # to handle the scenario when the agent keeps modifying without accepting a path
            reward = self._quality_score() - 0.2
            truncated = True
        
        obs = self._get_observation()
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Return observation space values"""
        target = self.target_duration
        actual = self.actual_duration
        diff = abs(target - actual)
        no_intr_nodes = len(self.current_loop) - 2                              # excluding the starting node and the end node
        episode_ratio = min(1.0, self.current_step / self.max_steps)            # ratio of completed steps to total steps
        overlap_ratio = self._route_overlap_ratio()                             # calculate the ratio of repeated nodes
        
        obs = np.array([target,
                        actual, 
                        diff,
                        no_intr_nodes,
                        episode_ratio,
                        overlap_ratio
                        ], dtype=np.float32)
        obs = np.clip(obs,                                                      # to bound the observation values
                    self.observation_space.low,
                    self.observation_space.high
                    ) 
        return obs
    
    def _generate_random_loop(self):
        """Create a random walkable path with a random number of 
        nodes near to the starting node."""
        num_nodes = int(self.np_random.integers(self.min_node, 7))                  # 7 because the maximum number of nodes is 10 and excluding the start and end node 
        candidates = self._candidate_nodes()                   
        indices = self.np_random.choice(len(candidates), num_nodes, replace=False)   # pick random unique candidate nodes
        nodes = [candidates[i] for i in indices]
        loop = [self.starting_node] + nodes + [self.starting_node]
        return loop

    def _add_node(self):
        """Insert one new node into the walkable path."""
        if len(self.current_loop) - 2 >= self.max_node:                            # checks if the current loop has reached the maximum number of nodes
            return

        new_node = self._get_new_nodes()                                           # gets new nodes
        if new_node is None:
            return

        insert_index = int(self.np_random.integers(1, len(self.current_loop)))     # insert the new node at a random index other than strting node
        self.current_loop.insert(insert_index, new_node)

    def _remove_node(self):
        """Remove one intermediate node while keeping the fixed start/end node."""
        if len(self.current_loop) - 2 <= self.min_node:                            # checks if the current loop has reached the minimum number of nodes
            return

        remove_index = int(self.np_random.integers(1, len(self.current_loop) - 1))  # remove the node at a random index other than start/ end node
        self.current_loop.pop(remove_index)

    def _replace_node(self):
        """Swap one existing intermediate node with a new unused node."""
        if len(self.current_loop) - 2 <= 0:                                        # checks if the current loop has nodes other than start/ end node
            self._add_node()
            return

        replace_index = int(self.np_random.integers(1, len(self.current_loop) - 1))  # replace the node at a random index other than start/ end node
        new_node = self._get_new_nodes(exclude_index=replace_index)                  # gets new nodes
        if new_node is None:
            return

        self.current_loop[replace_index] = new_node

    def _get_new_nodes(self, exclude_index=None):
        """Gets new nodes which are near to starting nodes and not used in the current loop."""
        used_nodes = set(self.current_loop)

        candidates = [node for node in self._candidate_nodes() if node not in used_nodes]
        if not candidates:
            return None

        return candidates[int(self.np_random.integers(0, len(candidates)))]         # gets a random new node and remove biases

    def _candidate_nodes(self):
        """Find node candidates at distances suitable for the target duration."""
        target_distance = self.target_duration * self.walking_speed     # target distance in meters calculated from target duration and walking speed
        cutoff = max(300, target_distance * 0.75)                       # 300 is an random number which is search radius from start node, and 0.75 is 75% of the target distance
        min_distance = max(50, target_distance * 0.08)                  # to avoid nodes closer than 50 meters near to the start node and 8% of the target distance
        max_distance = max(150, target_distance * 0.45)                 # maximum distance of the candidate node from the starting node

        lengths = nx.single_source_dijkstra_path_length(                # get the distance of each node from the starting node
            self.G,
            self.starting_node,
            cutoff=cutoff,
            weight='length'
        )
        candidates = [
            node for node, distance in lengths.items()                          # checks if the node distance is whithin the suitable range
            if node != self.starting_node and min_distance <= distance <= max_distance
        ]

        if len(candidates) >= self.max_node:
            return candidates

        return [node for node in self.nodes if node != self.starting_node]      # to handle scenarios if there are not enough candidate nodes

    def _quality_score(self):
        """Calculate the quality score of the current loop based on duration and overlap penalty."""
        diff = abs(self.target_duration - self.actual_duration)                 # calculate the absolute difference
        duration_score = max(0.0, 1.0 - (diff / self.target_duration))          # calculate the duration score, 1 means high quality
        overlap_penalty = self._route_overlap_ratio()                           # calculate the repeated nodes ratio
        return max(0.0, duration_score - overlap_penalty)                       # calculate the quality score

    def _build_route(self):
        """Build the route from the current loop."""
        route = []
        for i in range(len(self.current_loop) - 1):                             # calculate the distance between each pair of nodes
            u = self.current_loop[i]
            v = self.current_loop[i + 1]
            try:
                path = nx.shortest_path(self.G, u, v, weight='length')
                route.extend(path[:-1])                                         # exclude the last node
            except:
                return []

        if self.current_loop:
            route.append(self.current_loop[-1])                                 # add the last node to the final route

        return route

    def _route_overlap_ratio(self):
        """Calculate the fraction of repeated route nodes, ignoring start/end nodes."""
        route = self._build_route()
        if len(route) <= 2:                                                     # treat it as a bad loop, because only 2 nodes
            return 1.0

        route = route[:-1] if route[0] == route[-1] else route                  # exclude the end node because its same as the start node
        repeated_nodes = len(route) - len(set(route))                           # calculate the number of repeated nodes
        return min(1.0, repeated_nodes / len(route))                            # calculate the repeated nodes ratio, 0 means no repeated nodes

    def _compute_duration(self, loop):
        """Compute loop duration from shortest-path walking distance."""
        total_m = 0
        for i in range(len(loop) - 1):
            u = loop[i]
            v = loop[i + 1]
            try:
                meters = nx.shortest_path_length(self.G, u, v, weight='length')
                total_m += meters
            except:
                return 1000                                                         # if path is broken return heavy duration
        
        km = total_m / 1000.0                                                       # calculate duration with walking speed as 5km/h
        minutes = (km / 5.0) * 60.0
        return minutes
    
    def render(self):
        """Display the selected route map."""
        if self.render_mode == "human":
            route = self._build_route()
            if len(route) > 1:
                node_colors = ['red' if node == self.starting_node else 'white' for node in self.G.nodes]
                node_sizes = [80 if node == self.starting_node else 15 for node in self.G.nodes]
                ox.plot_graph_route(
                    self.G,
                    route,
                    route_color='r',
                    node_color=node_colors,
                    node_size=node_sizes,
                    show=True
                )
            else:
                print("No route found")
        

if __name__ == "__main__":
    env = gym.make('osm_map', render_mode='human')
    print("Check environment begin")
    check_env(env.unwrapped)                                                    # check whether the environment is valid
    print("Check environment end")
