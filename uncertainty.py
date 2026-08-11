
import random


class BayesianUncertaintyEngine:
    def __init__(self):
        self.high_traffic_risk_cells = set()

        # Prior Conditional Probabilities for the Bayesian Network
        self.probability_of_storm = 0.30
        self.probability_of_road_incident = 0.15

        # Conditional Probability Table: P(Congestion | Storm, Incident)
        self.cpt_congestion = {
            (True, True): 0.95,  # Storm present, Incident occurred
            (True, False): 0.70,  # Storm present, No incident
            (False, True): 0.85,  # Clear weather, Incident occurred
            (False, False): 0.20  # Clear weather, No incident
        }

    def register_high_traffic_zone(self, row_idx, col_idx):
        self.high_traffic_risk_cells.add((row_idx, col_idx))

    def unregister_high_traffic_zone(self, row_idx, col_idx):
        if (row_idx, col_idx) in self.high_traffic_risk_cells:
            self.high_traffic_risk_cells.remove((row_idx, col_idx))

    def clear_uncertainty_parameters(self):
        self.high_traffic_risk_cells.clear()

    def evaluate_congestion_probability(self):
        """Evaluates conditional dependencies across the network."""
        is_storming = random.random() < self.probability_of_storm
        has_incident = random.random() < self.probability_of_road_incident

        inference_probability = self.cpt_congestion[(is_storming, has_incident)]
        return inference_probability, is_storming, has_incident

    def calculate_dynamic_step_cost(self, initial_node, terminal_node):
        """Applies Bayesian probabilities to update edge weights."""
        base_step_cost = 1.0

        if terminal_node in self.high_traffic_risk_cells:
            congestion_prob, storm_flag, incident_flag = self.evaluate_congestion_probability()

            if congestion_prob > 0.50:
                # Stochastic delay cost scaling
                stochastic_delay_multiplier = random.uniform(2.5, 6.5)
                return base_step_cost + (congestion_prob * stochastic_delay_multiplier)

        return base_step_cost

    def calculate_accumulated_trajectory_cost(self, path_sequence):
        if not path_sequence:
            return 0.0

        aggregated_cost_sum = 0.0
        for idx in range(len(path_sequence) - 1):
            aggregated_cost_sum += self.calculate_dynamic_step_cost(path_sequence[idx], path_sequence[idx + 1])

        return aggregated_cost_sum