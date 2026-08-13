import unittest
import tkinter as tk
from gui import RoutePlannerCoreUI

class TestXAIDetailsAndLayout(unittest.TestCase):
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

    def test_search_caches_metrics_for_xai(self):
        self.ui.env.reconfigure_start_state(0, 0)
        self.ui.env.reconfigure_goal_state(2, 2)
        self.ui.csp.erase_all_constraints()

        # Initially, last_path should not be set
        self.assertFalse(hasattr(self.ui, 'last_path'))

        # Run A* search
        self.ui._run("A*")

        # Verify that performance metrics are successfully cached
        self.assertTrue(hasattr(self.ui, 'last_algo_id'))
        self.assertEqual(self.ui.last_algo_id, "A*")
        self.assertTrue(hasattr(self.ui, 'last_path'))
        self.assertGreater(len(self.ui.last_path), 0)
        self.assertTrue(hasattr(self.ui, 'last_explored'))
        self.assertGreater(len(self.ui.last_explored), 0)
        self.assertTrue(hasattr(self.ui, 'last_elapsed'))
        self.assertGreater(self.ui.last_elapsed, 0.0)
        self.assertTrue(hasattr(self.ui, 'last_cost'))

    def test_xai_natural_language_generation(self):
        self.ui.env.reconfigure_start_state(0, 0)
        self.ui.env.reconfigure_goal_state(2, 2)
        self.ui.csp.erase_all_constraints()

        self.ui._run("A*")

        # Get natural language explanation text
        explanation = self.ui.xai_text.get("1.0", tk.END).strip()
        self.assertIn("selected a route", explanation)
        self.assertIn("A*", explanation)
        self.assertIn("cost", explanation)
