class StructuralConstraintEngine:
    def __init__(self, baseline_financial_cap=350.0, baseline_temporal_cap=120.0):
        self.structural_obstacles_set = set()
        self.maximum_financial_budget = baseline_financial_cap
        self.maximum_time_limit = baseline_temporal_cap

    def register_impassable_obstacle(self, target_row, target_col):
        self.structural_obstacles_set.add((target_row, target_col))

    def remove_impassable_obstacle(self, target_row, target_col):
        if (target_row, target_col) in self.structural_obstacles_set:
            self.structural_obstacles_set.remove((target_row, target_col))

    def erase_all_constraints(self):
        self.structural_obstacles_set.clear()

    def assess_cell_viability(self, candidate_node):
        """Standard filtering node check."""
        # Returns True if the coordinate is unblocked and clear of obstructions
        return candidate_node not in self.structural_obstacles_set

    def evaluate_resource_compliance(self, calculated_financial_cost, computed_time_elapsed):
        """Verifies cumulative variables remain within defined thresholds."""
        if calculated_financial_cost > self.maximum_financial_budget:
            return False
        if computed_time_elapsed > self.maximum_time_limit:
            return False
        return True