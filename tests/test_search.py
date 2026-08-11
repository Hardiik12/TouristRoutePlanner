import unittest
from environment import SpatialEnvironment
import search

class TestSearchAlgorithms(unittest.TestCase):
    def setUp(self):
        # Initialize a default 10x10 environment for tests
        self.env = SpatialEnvironment(horizontal_dim=10, vertical_dim=10)
        self.default_validity = lambda coord: True
        self.default_cost = lambda c1, c2: 1.0

    def test_bfs_shortest_path_unit_cost(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 4)
        path, explored = search.execute_breadth_first_search(
            self.env, self.default_validity, self.default_cost
        )
        self.assertEqual(path, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
        self.assertIn((0, 0), explored)
        self.assertIn((0, 4), explored)

    def test_bfs_obstacle_handling(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        # Block direct path (0,1)
        validity = lambda coord: coord != (0, 1)
        path, _ = search.execute_breadth_first_search(
            self.env, validity, self.default_cost
        )
        # Path must detour around (0,1)
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)])

    def test_bfs_boundary_handling(self):
        self.env.reconfigure_start_state(0, 0)
        # Verify valid neighbors are constrained to boundaries
        neighbors = self.env.fetch_valid_cardinal_neighbors((0, 0))
        self.assertEqual(sorted(neighbors), [(0, 1), (1, 0)])

    def test_bfs_start_equals_goal(self):
        self.env.reconfigure_start_state(2, 2)
        self.env.reconfigure_goal_state(2, 2)
        path, explored = search.execute_breadth_first_search(
            self.env, self.default_validity, self.default_cost
        )
        self.assertEqual(path, [(2, 2)])
        self.assertEqual(explored, [(2, 2)])

    def test_bfs_no_path(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        # Surround start completely with obstacles
        validity = lambda coord: coord not in [(0, 1), (1, 0)]
        path, _ = search.execute_breadth_first_search(
            self.env, validity, self.default_cost
        )
        self.assertEqual(path, [])

    def test_dfs_valid_path_exists(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(2, 2)
        path, explored = search.execute_depth_first_search(
            self.env, self.default_validity, self.default_cost
        )
        self.assertTrue(len(path) >= 5) # Minimum Manhattan steps
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 2))
        self.assertIn((0, 0), explored)

    def test_dfs_obstacle_handling(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        validity = lambda coord: coord != (0, 1)
        path, _ = search.execute_depth_first_search(
            self.env, validity, self.default_cost
        )
        self.assertTrue(len(path) > 0)
        self.assertNotIn((0, 1), path)

    def test_dfs_no_path(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        validity = lambda coord: coord not in [(0, 1), (1, 0)]
        path, _ = search.execute_depth_first_search(
            self.env, validity, self.default_cost
        )
        self.assertEqual(path, [])

    def test_ucs_optimal_weighted_path(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        # Direct path (0,0) -> (0,1) -> (0,2) has step costs of 10.0 each (total = 20.0)
        # Detour path (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2) has step costs of 1.0 each (total = 4.0)
        def custom_cost(c1, c2):
            if c2 in [(0, 1), (0, 2)] and c1 in [(0, 0), (0, 1)]:
                return 10.0
            return 1.0

        path, _ = search.execute_uniform_cost_search(
            self.env, self.default_validity, custom_cost
        )
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)])

    def test_ucs_stale_entry_regression(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        
        # We set up a scenario where (0,1) can be reached:
        # 1. Directly from (0,0) with cost 100.0 (shorter in steps, higher cost)
        # 2. Detour via (1,0) -> (1,1) -> (0,1) with costs 1.0 + 1.0 + 1.0 = 3.0 (longer in steps, lower cost)
        # Then, from (0,1), we can reach the goal (0,2) with cost 1.0.
        # UCS should correctly use the lower cost path to (0,1) and successfully find the path of cost 4.0.
        def custom_cost(c1, c2):
            if c1 == (0, 0) and c2 == (0, 1):
                return 100.0
            return 1.0

        path, explored = search.execute_uniform_cost_search(
            self.env, self.default_validity, custom_cost
        )
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])
        
        # Verify that (0,1) is only expanded once inexplored history, or the stale expansion was skipped
        # The stale entry (popped with cost 100.0) should have been skipped.
        # So (0,1)'s neighbors should NOT be expanded a second time.
        # We can check that the total explored node count does not contain duplicates of (0,1) expanded after goal.
        goal_index = explored.index((0, 2))
        post_goal_explored = explored[goal_index + 1:]
        self.assertNotIn((0, 1), post_goal_explored)

    def test_astar_optimal_weighted_path(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        def custom_cost(c1, c2):
            if c2 in [(0, 1), (0, 2)] and c1 in [(0, 0), (0, 1)]:
                return 10.0
            return 1.0

        path, _ = search.execute_astar_search(
            self.env, self.default_validity, custom_cost
        )
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)])

    def test_astar_stale_entry_regression(self):
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(0, 2)
        def custom_cost(c1, c2):
            if c1 == (0, 0) and c2 == (0, 1):
                return 100.0
            return 1.0

        path, explored = search.execute_astar_search(
            self.env, self.default_validity, custom_cost
        )
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])
        
        # Ensure stale A* queue entries did not cause duplicate optimal-node neighbor expansions
        goal_index = explored.index((0, 2))
        post_goal_explored = explored[goal_index + 1:]
        self.assertNotIn((0, 1), post_goal_explored)

    def test_cross_validation_unit_cost(self):
        # On a clear, unit-cost grid, BFS, UCS, and A* should produce the exact same path cost
        self.env.reconfigure_start_state(0, 0)
        self.env.reconfigure_goal_state(5, 5)
        
        bfs_path, _ = search.execute_breadth_first_search(
            self.env, self.default_validity, self.default_cost
        )
        ucs_path, _ = search.execute_uniform_cost_search(
            self.env, self.default_validity, self.default_cost
        )
        astar_path, _ = search.execute_astar_search(
            self.env, self.default_validity, self.default_cost
        )
        
        self.assertEqual(len(bfs_path), len(ucs_path))
        self.assertEqual(len(astar_path), len(ucs_path))
        self.assertEqual(len(bfs_path), 11) # 10 steps (start to goal is 5+5=10 transitions)

    def test_gui_validation_checks(self):
        import tkinter as tk
        from gui import RoutePlannerCoreUI

        try:
            root = tk.Tk()
            root.withdraw() # Hide the window
        except (tk.TclError, RuntimeError):
            self.skipTest("No GUI environment available")
            return

        try:
            ui = RoutePlannerCoreUI(root)

            # Test 1: Start equals Goal
            ui.env.reconfigure_start_state(2, 2)
            ui.env.reconfigure_goal_state(2, 2)
            valid, msg = ui._validate_planning_state()
            self.assertFalse(valid)
            self.assertEqual(msg, "Already at destination!")

            # Test 2: Start blocked by obstacle
            ui.env.reconfigure_start_state(0, 0)
            ui.env.reconfigure_goal_state(0, 2)
            ui.csp.register_impassable_obstacle(0, 0)
            valid, msg = ui._validate_planning_state()
            self.assertFalse(valid)
            self.assertEqual(msg, "Start blocked by obstacle!")

            # Clear obstacles and verify valid
            ui.csp.erase_all_constraints()
            valid, msg = ui._validate_planning_state()
            self.assertTrue(valid)
        finally:
            root.destroy()

    def test_gui_animation_speed_delay(self):
        import tkinter as tk
        from gui import RoutePlannerCoreUI

        try:
            root = tk.Tk()
            root.withdraw()
        except (tk.TclError, RuntimeError):
            self.skipTest("No GUI environment available")
            return

        try:
            ui = RoutePlannerCoreUI(root)

            # Test default / Normal delay values
            ui.speed_var.set("Normal")
            self.assertEqual(ui._get_animation_delay("explore"), 6)
            self.assertEqual(ui._get_animation_delay("path"), 28)
            self.assertEqual(ui._get_animation_delay("robot"), 40)

            # Test Instant delay values (0)
            ui.speed_var.set("Instant")
            self.assertEqual(ui._get_animation_delay("explore"), 0)
            self.assertEqual(ui._get_animation_delay("path"), 0)
            self.assertEqual(ui._get_animation_delay("robot"), 0)

            # Test Fast delay values (0.25x multiplier)
            ui.speed_var.set("Fast")
            self.assertEqual(ui._get_animation_delay("explore"), int(6 * 0.25))
            self.assertEqual(ui._get_animation_delay("path"), int(28 * 0.25))
            self.assertEqual(ui._get_animation_delay("robot"), int(40 * 0.25))

            # Test Slow delay values (3.0x multiplier)
            ui.speed_var.set("Slow")
            self.assertEqual(ui._get_animation_delay("explore"), 6 * 3)
            self.assertEqual(ui._get_animation_delay("path"), 28 * 3)
            self.assertEqual(ui._get_animation_delay("robot"), 40 * 3)
        finally:
            root.destroy()



