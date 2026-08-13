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

    def test_bayesian_sliders_update_congestion_inference(self):
        # Set storm and road incident priors to high
        self.ui.uncertainty.probability_of_storm = 0.90
        self.ui.uncertainty.probability_of_road_incident = 0.85
        
        self.ui._update_bayes_inference()
        
        # Verify that P(Storm) and P(Incident) text labels display values
        self.assertEqual(self.ui.b_storm.cget("text"), "90%")
        self.assertEqual(self.ui.b_incident.cget("text"), "85%")
        
        # Verify that P(Congestion) calculates a high probability
        congest_prob_str = self.ui.b_congest.cget("text")
        self.assertTrue(congest_prob_str.endswith("%"))
        self.assertGreaterEqual(int(congest_prob_str.replace("%", "")), 50)

    def test_random_obstacles_population(self):
        self.ui.obstacle_density_var.set(0.15)  # 15% density
        self.ui.env.reconfigure_start_state(0, 0)
        self.ui.env.reconfigure_goal_state(19, 19)
        
        self.ui._generate_random_obstacles()
        
        obstacles = self.ui.csp.structural_obstacles_set
        self.assertGreater(len(obstacles), 0)
        # Check start and goal coordinates are free from obstacles
        self.assertNotIn((0, 0), obstacles)
        self.assertNotIn((19, 19), obstacles)

    def test_stitched_sequential_tour_path(self):
        self.ui.env.reconfigure_start_state(5, 5)
        self.ui.tour_mode_var.set(True)
        
        # Trigger landmarks journey (Start -> Beach -> Museum -> Temple -> Park -> Mall)
        # Beach is at (0,0), Museum (3,5), Temple (10,12), Park (14,8), Mall (19,19)
        self.ui.route_algo_var.set("A*")
        
        # Call the routing helper
        self.ui.route_landmarks_fn()
        
        # Verify that the path starts at (5,5), visits milestones, and reaches Mall at (19,19)
        self.assertTrue(hasattr(self.ui, 'last_path'))
        self.assertEqual(self.ui.last_path[0], (5, 5))
        self.assertEqual(self.ui.last_path[-1], (19, 19))
        self.assertIn((0, 0), self.ui.last_path)
        self.assertIn((3, 5), self.ui.last_path)
        self.assertIn((10, 12), self.ui.last_path)
        self.assertIn((14, 8), self.ui.last_path)

