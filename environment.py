
class SpatialEnvironment:
    def __init__(self, horizontal_dim=20, vertical_dim=20):
        self.total_rows = horizontal_dim
        self.total_cols = vertical_dim
        self.start_coordinate = (0, 0)
        self.goal_coordinate = (horizontal_dim - 1, vertical_dim - 1)
        self.urban_landmarks = {}
        self.initialize_default_landmarks()

    def initialize_default_landmarks(self):
        """Maps specific coordinates to human-readable tourist landmarks."""
        # Baseline configurations based on user mapping preferences
        self.urban_landmarks[(0, 0)] = "Beach"
        self.urban_landmarks[(3, 5)] = "Museum"
        self.urban_landmarks[(10, 12)] = "Temple"
        self.urban_landmarks[(14, 8)] = "Park"
        self.urban_landmarks[(self.total_rows - 1, self.total_cols - 1)] = "Mall"

    def reconfigure_start_state(self, row_idx, col_idx):
        if 0 <= row_idx < self.total_rows and 0 <= col_idx < self.total_cols:
            self.start_coordinate = (row_idx, col_idx)

    def reconfigure_goal_state(self, row_idx, col_idx):
        if 0 <= row_idx < self.total_rows and 0 <= col_idx < self.total_cols:
            self.goal_coordinate = (row_idx, col_idx)

    def fetch_valid_cardinal_neighbors(self, evaluated_node):
        """Enforces a strict 4-way cardinal connectivity rule."""
        current_row, current_col = evaluated_node
        potential_neighbors = []

        # Matrix translation directions: Up, Down, Left, Right
        cardinal_translations = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row_offset, col_offset in cardinal_translations:
            target_row = current_row + row_offset
            target_col = current_col + col_offset

            # Verify coordinates fall within structural boundaries
            if 0 <= target_row < self.total_rows and 0 <= target_col < self.total_cols:
                potential_neighbors.append((target_row, target_col))

        return potential_neighbors

    def resolve_landmark_identity(self, node_coordinate):
        return self.urban_landmarks.get(node_coordinate, f"Waypoint ({node_coordinate[0]},{node_coordinate[1]})")