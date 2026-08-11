
"""
Graph Traversal Computational Core.
Implements the algorithmic state space routing routines
for BFS, DFS, UCS, and Informed Heuristic A* Search.
"""

from queue import Queue, PriorityQueue


class CodebaseStack:
    """Emulates stack behavior using lists to support LIFO data routines."""

    def __init__(self):
        self.internal_storage = []

    def push_element(self, item):
        self.internal_storage.append(item)

    def pop_element(self):
        return self.internal_storage.pop() if not self.is_empty() else None

    def is_empty(self):
        return len(self.internal_storage) == 0


def execute_breadth_first_search(env_graph, validity_evaluator, step_cost_calculator):
    origin = env_graph.start_coordinate
    target = env_graph.goal_coordinate

    execution_frontier = Queue()
    execution_frontier.put(origin)

    navigation_ancestry = {origin: None}
    nodes_explored_history = []

    while not execution_frontier.empty():
        active_node = execution_frontier.get()
        nodes_explored_history.append(active_node)

        if active_node == target:
            break

        for neighbor in env_graph.fetch_valid_cardinal_neighbors(active_node):
            if validity_evaluator(neighbor) and neighbor not in navigation_ancestry:
                navigation_ancestry[neighbor] = active_node
                execution_frontier.put(neighbor)

    return compile_reconstructed_path(navigation_ancestry, origin, target), nodes_explored_history


def execute_depth_first_search(env_graph, validity_evaluator, step_cost_calculator):
    origin = env_graph.start_coordinate
    target = env_graph.goal_coordinate

    execution_frontier = CodebaseStack()
    execution_frontier.push_element(origin)

    navigation_ancestry = {origin: None}
    nodes_explored_history = []
    nodes_visited_set = set()

    while not execution_frontier.is_empty():
        active_node = execution_frontier.pop_element()

        if active_node in nodes_visited_set:
            continue

        nodes_visited_set.add(active_node)
        nodes_explored_history.append(active_node)

        if active_node == target:
            break

        for neighbor in env_graph.fetch_valid_cardinal_neighbors(active_node):
            if validity_evaluator(neighbor) and neighbor not in nodes_visited_set:
                navigation_ancestry[neighbor] = active_node
                execution_frontier.push_element(neighbor)

    return compile_reconstructed_path(navigation_ancestry, origin, target), nodes_explored_history


def execute_uniform_cost_search(env_graph, validity_evaluator, step_cost_calculator):
    origin = env_graph.start_coordinate
    target = env_graph.goal_coordinate

    execution_frontier = PriorityQueue()
    execution_frontier.put((0.0, origin))

    navigation_ancestry = {origin: None}
    cumulative_costs = {origin: 0.0}
    nodes_explored_history = []

    while not execution_frontier.empty():
        active_cost, active_node = execution_frontier.get()

        # Stale-entry protection: if a better route to this node was already processed, skip
        if active_cost > cumulative_costs.get(active_node, float('inf')):
            continue

        nodes_explored_history.append(active_node)

        if active_node == target:
            break

        for neighbor in env_graph.fetch_valid_cardinal_neighbors(active_node):
            if not validity_evaluator(neighbor):
                continue

            incremental_cost = step_cost_calculator(active_node, neighbor)
            calculated_cost = cumulative_costs[active_node] + incremental_cost

            if neighbor not in cumulative_costs or calculated_cost < cumulative_costs[neighbor]:
                cumulative_costs[neighbor] = calculated_cost
                navigation_ancestry[neighbor] = active_node
                execution_frontier.put((calculated_cost, neighbor))

    return compile_reconstructed_path(navigation_ancestry, origin, target), nodes_explored_history


def execute_astar_search(env_graph, validity_evaluator, step_cost_calculator):
    origin = env_graph.start_coordinate
    target = env_graph.goal_coordinate

    execution_frontier = PriorityQueue()
    initial_heuristic = abs(origin[0] - target[0]) + abs(origin[1] - target[1])
    execution_frontier.put((initial_heuristic, 0.0, origin))

    navigation_ancestry = {origin: None}
    cumulative_costs = {origin: 0.0}
    nodes_explored_history = []

    while not execution_frontier.empty():
        _, active_g, active_node = execution_frontier.get()

        # Stale-entry protection: if a better route to this node was already processed, skip
        if active_g > cumulative_costs.get(active_node, float('inf')):
            continue

        nodes_explored_history.append(active_node)

        if active_node == target:
            break

        for neighbor in env_graph.fetch_valid_cardinal_neighbors(active_node):
            if not validity_evaluator(neighbor):
                continue

            incremental_cost = step_cost_calculator(active_node, neighbor)
            calculated_cost = cumulative_costs[active_node] + incremental_cost

            if neighbor not in cumulative_costs or calculated_cost < cumulative_costs[neighbor]:
                cumulative_costs[neighbor] = calculated_cost

                # Admissible Manhattan Distance Heuristic Calculation
                heuristic_estimate = abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])
                evaluation_function_score = calculated_cost + heuristic_estimate

                navigation_ancestry[neighbor] = active_node
                execution_frontier.put((evaluation_function_score, calculated_cost, neighbor))

    return compile_reconstructed_path(navigation_ancestry, origin, target), nodes_explored_history


def compile_reconstructed_path(ancestry_map, origin, target):
    if target not in ancestry_map:
        return []

    active_trace_node = target
    compiled_trajectory = []

    while active_trace_node is not None:
        compiled_trajectory.append(active_trace_node)
        active_trace_node = ancestry_map[active_trace_node]

    compiled_trajectory.reverse()
    return compiled_trajectory