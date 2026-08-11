
class SequentialDecisionMatrix:
    def __init__(self):
        self.execution_analytics_matrix = {}

    def catalog_algorithm_performance(self, algorithm_id, complete_path, total_explored, aggregated_cost):
        self.execution_analytics_matrix[algorithm_id] = {
            "path_sequence": complete_path,
            "nodes_expanded_count": len(total_explored),
            "total_computed_cost": aggregated_cost,
            "viability_status": len(complete_path) > 0
        }

    def execute_minimax_lookahead(self, active_node, target_depth, maximize_player, env_ref, cost_lookup_fn):
        """
        Executes a multi-tier search to evaluate path durability
        against dynamic traffic cost increases.
        """
        if target_depth == 0 or active_node == env_ref.goal_coordinate:
            # Base return value: absolute coordinate validation evaluation
            return abs(active_node[0] - env_ref.goal_coordinate[0]) + abs(active_node[1] - env_ref.goal_coordinate[1])

        valid_neighbors = env_ref.fetch_valid_cardinal_neighbors(active_node)
        if not valid_neighbors:
            return float('inf')

        if maximize_player:
            maximum_evaluation = float('-inf')
            for neighbor in valid_neighbors:
                evaluation_score = self.execute_minimax_lookahead(neighbor, target_depth - 1, False, env_ref,
                                                                  cost_lookup_fn)
                maximum_evaluation = max(maximum_evaluation, evaluation_score)
            return maximum_evaluation
        else:
            minimum_evaluation = float('inf')
            for neighbor in valid_neighbors:
                # The environment applies simulated traffic increases
                added_delay_penalty = cost_lookup_fn(active_node, neighbor)
                evaluation_score = self.execute_minimax_lookahead(neighbor, target_depth - 1, True, env_ref,
                                                                  cost_lookup_fn) + added_delay_penalty
                minimum_evaluation = min(minimum_evaluation, evaluation_score)
            return minimum_evaluation

    def isolate_optimal_engineered_route(self, budget_limit_threshold):
        """Identifies the optimal route from the execution matrix."""
        prime_selected_algorithm = None
        lowest_observed_cost = float('inf')

        for algorithm_id, performance_dataset in self.execution_analytics_matrix.items():
            if not performance_dataset["viability_status"]:
                continue
            if performance_dataset["total_computed_cost"] > budget_limit_threshold:
                continue

            if performance_dataset["total_computed_cost"] < lowest_observed_cost:
                lowest_observed_cost = performance_dataset["total_computed_cost"]
                prime_selected_algorithm = algorithm_id

        return prime_selected_algorithm, self.execution_analytics_matrix.get(prime_selected_algorithm, None)

    def reset_decision_matrix(self):
        self.execution_analytics_matrix.clear()


class ExplainablePipelineEngine:
    def __init__(self, spatial_env, constraint_unit, uncertainty_unit):
        self.env = spatial_env
        self.csp = constraint_unit
        self.uncertainty = uncertainty_unit

    def construct_natural_language_explanation(self, chosen_algo, path_sequence, node_expansions_count,
                                               calculated_cost):
        if not path_sequence:
            return "Routing Execution Failed. Reason: Target destination is completely blocked by physical obstacles."

        origin_landmark = self.env.resolve_landmark_identity(path_sequence[0])
        terminal_landmark = self.env.resolve_landmark_identity(path_sequence[-1])

        # Analyze path composition to count traffic intersections
        congested_cells_crossed = 0
        for node in path_sequence:
            if node in self.uncertainty.high_traffic_risk_cells:
                congested_cells_crossed += 1

        # Build structural reasoning explanations based on performance characteristics
        explanation_segments = [
            f"The System selected a route from '{origin_landmark}' to '{terminal_landmark}' using {chosen_algo} Search optimization.",
            f"The final path requires {len(path_sequence)} step movements, expanding {node_expansions_count} nodes to confirm path safety.",
            f"The total path cost evaluates to {calculated_cost:.2f} resource units."
        ]

        if congested_cells_crossed > 0:
            explanation_segments.append(
                f"The route crosses {congested_cells_crossed} known high-traffic areas, accepting controlled delays to avoid long physical detours."
            )
        else:
            explanation_segments.append(
                "The path successfully routed around all active traffic zones, prioritizing a clear route over immediate geometric paths."
            )

        if calculated_cost < (self.csp.maximum_financial_budget * 0.5):
            explanation_segments.append(
                "Resource usage remains highly efficient, conserving over half of the allocated travel budget.")
        else:
            explanation_segments.append(
                "The path complies with budget safety limits, but requires higher resource use due to current traffic conditions.")

        return " ".join(explanation_segments)