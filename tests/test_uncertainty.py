import unittest
import random
from uncertainty import BayesianUncertaintyEngine

class TestBayesianUncertainty(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianUncertaintyEngine()

    def test_priors_initialization(self):
        # Verify default prior probabilities match specifications
        self.assertEqual(self.engine.probability_of_storm, 0.30)
        self.assertEqual(self.engine.probability_of_road_incident, 0.15)

    def test_probability_bounds_and_types(self):
        for _ in range(100):
            prob, storm, incident = self.engine.evaluate_congestion_probability()
            self.assertTrue(0.0 <= prob <= 1.0)
            self.assertIsInstance(storm, bool)
            self.assertIsInstance(incident, bool)

    def test_congestion_probability_values(self):
        valid_cpt_values = {0.20, 0.70, 0.85, 0.95}
        for _ in range(100):
            prob, _, _ = self.engine.evaluate_congestion_probability()
            self.assertIn(prob, valid_cpt_values)

    def test_high_traffic_registration(self):
        cell = (3, 3)
        self.engine.register_high_traffic_zone(*cell)
        self.assertIn(cell, self.engine.high_traffic_risk_cells)

        self.engine.unregister_high_traffic_zone(*cell)
        self.assertNotIn(cell, self.engine.high_traffic_risk_cells)

    def test_dynamic_step_cost_free_cell(self):
        # Non-traffic cells must have a deterministic base cost of 1.0
        self.assertEqual(self.engine.calculate_dynamic_step_cost((0, 0), (1, 1)), 1.0)

    def test_dynamic_step_cost_mathematical_invariants(self):
        cell = (2, 2)
        self.engine.register_high_traffic_zone(*cell)

        # Save original method
        original_evaluate = self.engine.evaluate_congestion_probability

        try:
            # 1. Test when congestion probability is <= 0.50
            self.engine.evaluate_congestion_probability = lambda: (0.20, False, False)
            for _ in range(10):
                cost = self.engine.calculate_dynamic_step_cost((0, 0), cell)
                self.assertEqual(cost, 1.0)

            # 2. Test when congestion probability is > 0.50 (e.g., 0.70)
            self.engine.evaluate_congestion_probability = lambda: (0.70, True, False)
            for _ in range(100):
                cost = self.engine.calculate_dynamic_step_cost((0, 0), cell)
                # Cost bounds: 1.0 + 0.70 * 2.5 <= cost <= 1.0 + 0.70 * 6.5
                min_bound = 1.0 + 0.70 * 2.5
                max_bound = 1.0 + 0.70 * 6.5
                self.assertTrue(min_bound <= cost <= max_bound, f"Cost {cost} out of bounds [{min_bound}, {max_bound}] for prob 0.70")
        finally:
            # Restore original method
            self.engine.evaluate_congestion_probability = original_evaluate

    def test_deterministic_seeding(self):
        cell = (2, 2)
        self.engine.register_high_traffic_zone(*cell)

        # Seed random to ensure exact matching behavior
        random.seed(42)
        cost_1 = self.engine.calculate_dynamic_step_cost((0, 0), cell)

        random.seed(42)
        cost_2 = self.engine.calculate_dynamic_step_cost((0, 0), cell)

        self.assertEqual(cost_1, cost_2)

    def test_accumulated_trajectory_cost_empty_path(self):
        self.assertEqual(self.engine.calculate_accumulated_trajectory_cost([]), 0.0)
        self.assertEqual(self.engine.calculate_accumulated_trajectory_cost([(0, 0)]), 0.0)

    def test_accumulated_trajectory_cost_unit_cost(self):
        # Obstacle-free/traffic-free path of length 5 (4 transitions) must evaluate to 4.0
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        self.assertEqual(self.engine.calculate_accumulated_trajectory_cost(path), 4.0)
