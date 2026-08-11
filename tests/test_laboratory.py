import unittest
import tkinter as tk
from gui import RoutePlannerCoreUI

class TestAlgorithmLaboratory(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
            self.root.withdraw()
            self.ui = RoutePlannerCoreUI(self.root)
        except (tk.TclError, RuntimeError):
            self.skipTest("No GUI environment available")

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def test_comparison_calculations_and_data_saving(self):
        # Setup clear coordinates
        self.ui.env.reconfigure_start_state(0, 0)
        self.ui.env.reconfigure_goal_state(3, 3)
        self.ui.csp.erase_all_constraints()

        # Run comparison laboratory analysis
        self.ui._run_lab_comparison()

        # Check results are populated in comparison_results dictionary
        self.assertIn("BFS", self.ui.comparison_results)
        self.assertIn("DFS", self.ui.comparison_results)
        self.assertIn("UCS", self.ui.comparison_results)
        self.assertIn("A*", self.ui.comparison_results)

        # Check metrics format and correctness
        for aid in ["BFS", "DFS", "UCS", "A*"]:
            res = self.ui.comparison_results[aid]
            self.assertEqual(res["status"], "Found")
            self.assertNotEqual(res["cost"], "-")
            self.assertIn("ms", res["time"])

        # Test failure/no path scenario
        # Block the goal completely with obstacles (4 cardinal directions for (3,3))
        self.ui.csp.register_impassable_obstacle(2, 3) # Up
        self.ui.csp.register_impassable_obstacle(3, 2) # Left
        self.ui.csp.register_impassable_obstacle(4, 3) # Down
        self.ui.csp.register_impassable_obstacle(3, 4) # Right

        self.ui._run_lab_comparison()

        # Check that status updates to "No Path"
        for aid in ["BFS", "DFS", "UCS", "A*"]:
            res = self.ui.comparison_results[aid]
            self.assertEqual(res["status"], "No Path")
            self.assertEqual(res["cost"], "-")
