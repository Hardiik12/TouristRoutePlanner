import unittest
from decision import SequentialDecisionMatrix
from environment import SpatialEnvironment

class TestDecisionMinimax(unittest.TestCase):
    def setUp(self):
        self.decision_matrix = SequentialDecisionMatrix()
        self.env = SpatialEnvironment(horizontal_dim=5, vertical_dim=5)
        self.env.reconfigure_goal_state(4, 4)
        self.default_cost = lambda c1, c2: 1.0

    def test_minimax_terminal_at_goal(self):
        # At the goal, minimax score must be 0 regardless of depth or player
        self.env.reconfigure_goal_state(2, 2)
        score = self.decision_matrix.execute_minimax_lookahead(
            (2, 2), target_depth=3, maximize_player=True, env_ref=self.env, cost_lookup_fn=self.default_cost
        )
        self.assertEqual(score, 0.0)

    def test_minimax_terminal_depth_zero(self):
        # At depth 0, minimax score must be the Manhattan distance to the goal
        score = self.decision_matrix.execute_minimax_lookahead(
            (0, 0), target_depth=0, maximize_player=True, env_ref=self.env, cost_lookup_fn=self.default_cost
        )
        self.assertEqual(score, 8.0) # |0-4| + |0-4| = 8

    def test_minimax_maximizing_player(self):
        # From (0,0), neighbors are (0,1) [distance 7] and (1,0) [distance 7]
        # At depth 1, maximize_player=True will return the max of their terminal values
        score = self.decision_matrix.execute_minimax_lookahead(
            (0, 0), target_depth=1, maximize_player=True, env_ref=self.env, cost_lookup_fn=self.default_cost
        )
        # Neighbors evaluated at depth 0:
        # (0,1) -> distance to (4,4) is 7. (1,0) -> distance to (4,4) is 7.
        # Max is 7.
        self.assertEqual(score, 7.0)

    def test_minimax_minimizing_player(self):
        # From (0,0), neighbors are (0,1) and (1,0).
        # We define a custom cost function where moving to (0,1) is extremely expensive (100.0)
        # and moving to (1,0) is cheap (1.0).
        # At depth 1, minimize_player=False will add step cost to the evaluated score:
        # transition to (0,1) -> score = 7.0 + 100.0 = 107.0
        # transition to (1,0) -> score = 7.0 + 1.0 = 8.0
        # Min of these is 8.0.
        def custom_cost(c1, c2):
            if c2 == (0, 1):
                return 100.0
            return 1.0

        score = self.decision_matrix.execute_minimax_lookahead(
            (0, 0), target_depth=1, maximize_player=False, env_ref=self.env, cost_lookup_fn=custom_cost
        )
        self.assertEqual(score, 8.0)

    def test_isolate_optimal_route_selection(self):
        # Catalog performance metrics for different algorithms
        self.decision_matrix.catalog_algorithm_performance("BFS", [(0,0), (0,1)], [(0,0)], 80.0)
        self.decision_matrix.catalog_algorithm_performance("A*", [(0,0), (0,1)], [(0,0)], 50.0)
        self.decision_matrix.catalog_algorithm_performance("UCS", [(0,0), (0,1)], [(0,0)], 30.0)
        self.decision_matrix.catalog_algorithm_performance("DFS", [(0,0), (0,1)], [(0,0)], 120.0)

        # 1. Under budget limit of 100.0, UCS should be selected as the cheapest path
        best_algo, metrics = self.decision_matrix.isolate_optimal_engineered_route(budget_limit_threshold=100.0)
        self.assertEqual(best_algo, "UCS")
        self.assertEqual(metrics["total_computed_cost"], 30.0)

        # 2. Under budget limit of 40.0, UCS is still the only one within budget
        best_algo, _ = self.decision_matrix.isolate_optimal_engineered_route(budget_limit_threshold=40.0)
        self.assertEqual(best_algo, "UCS")

        # 3. Under budget limit of 20.0, no path is within budget
        best_algo, metrics = self.decision_matrix.isolate_optimal_engineered_route(budget_limit_threshold=20.0)
        self.assertIsNone(best_algo)
        self.assertIsNone(metrics)

    def test_reset_decision_matrix(self):
        self.decision_matrix.catalog_algorithm_performance("A*", [(0,0), (0,1)], [(0,0)], 50.0)
        self.assertEqual(len(self.decision_matrix.execution_analytics_matrix), 1)

        self.decision_matrix.reset_decision_matrix()
        self.assertEqual(len(self.decision_matrix.execution_analytics_matrix), 0)
