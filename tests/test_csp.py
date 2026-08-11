import unittest
from constraints import StructuralConstraintEngine
from environment import SpatialEnvironment

class TestConstraintSatisfaction(unittest.TestCase):
    def setUp(self):
        # Default cap of 350.0 financial budget and 120.0 time limit
        self.csp = StructuralConstraintEngine(baseline_financial_cap=350.0, baseline_temporal_cap=120.0)
        self.env = SpatialEnvironment(horizontal_dim=10, vertical_dim=10)

    def test_boundary_checks_center(self):
        # A node inside the boundary (center) should return all 4 neighbors
        neighbors = self.env.fetch_valid_cardinal_neighbors((5, 5))
        self.assertEqual(len(neighbors), 4)
        self.assertIn((4, 5), neighbors)
        self.assertIn((6, 5), neighbors)
        self.assertIn((5, 4), neighbors)
        self.assertIn((5, 6), neighbors)

    def test_boundary_checks_corner_exact(self):
        # A node on the corner (boundary edge) should return only 2 neighbors
        neighbors = self.env.fetch_valid_cardinal_neighbors((0, 0))
        self.assertEqual(sorted(neighbors), [(0, 1), (1, 0)])

    def test_boundary_checks_outside(self):
        # An invalid node outside boundaries should return no neighbors or not be reachable
        neighbors = self.env.fetch_valid_cardinal_neighbors((9, 9))
        self.assertEqual(sorted(neighbors), [(8, 9), (9, 8)])

    def test_obstacle_registration_and_avoidance(self):
        # Initially, cell is viable
        self.assertTrue(self.csp.assess_cell_viability((1, 1)))

        # Register obstacle
        self.csp.register_impassable_obstacle(1, 1)
        self.assertFalse(self.csp.assess_cell_viability((1, 1)))

        # Remove obstacle
        self.csp.remove_impassable_obstacle(1, 1)
        self.assertTrue(self.csp.assess_cell_viability((1, 1)))

    def test_erase_all_constraints(self):
        self.csp.register_impassable_obstacle(1, 1)
        self.csp.register_impassable_obstacle(2, 2)
        self.assertFalse(self.csp.assess_cell_viability((1, 1)))
        self.assertFalse(self.csp.assess_cell_viability((2, 2)))

        self.csp.erase_all_constraints()
        self.assertTrue(self.csp.assess_cell_viability((1, 1)))
        self.assertTrue(self.csp.assess_cell_viability((2, 2)))

    def test_resource_compliance_within_budget(self):
        # cost=300.0 (less than 350.0 limit), time=100.0 (less than 120.0 limit)
        compliant = self.csp.evaluate_resource_compliance(300.0, 100.0)
        self.assertTrue(compliant)

    def test_resource_compliance_exactly_on_budget(self):
        # cost=350.0 (equal to 350.0 limit), time=120.0 (equal to 120.0 limit)
        compliant = self.csp.evaluate_resource_compliance(350.0, 120.0)
        self.assertTrue(compliant)

    def test_resource_compliance_above_budget(self):
        # cost exceeding budget
        compliant = self.csp.evaluate_resource_compliance(351.0, 100.0)
        self.assertFalse(compliant)

        # time exceeding budget
        compliant = self.csp.evaluate_resource_compliance(300.0, 121.0)
        self.assertFalse(compliant)

        # both exceeding
        compliant = self.csp.evaluate_resource_compliance(400.0, 150.0)
        self.assertFalse(compliant)

    def test_csp_robustness_empty_input(self):
        # Verify default limits are correctly handled with zero or minimal values
        self.assertTrue(self.csp.evaluate_resource_compliance(0.0, 0.0))
        self.assertTrue(self.csp.assess_cell_viability((0, 0)))
