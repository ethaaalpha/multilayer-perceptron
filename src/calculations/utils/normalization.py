class Normalizer:
    """Normalization of the value between [0, 1] using min-max, if already between 0 and 1, skipping"""

    def normalize_list(self, data: list[tuple[float]]) -> list[tuple[float]]:
        min_t, max_t = min(data), max(data)
        all_t_normalized = [self.normalize(min_t, max_t, t) for t in data]

        return all_t_normalized

    # def denormalize_list(self, original_data: list[tuple[float]], normalized_data: list[tuple[float]]) -> list[tuple[float]]:
    #     min_t, max_t = min(original_data), max(original_data)
    #     all_t_reverted = [self.denormalize(min_t, max_t, t) for t in normalized_data]
        
    #     return all_t_reverted

    def normalize(self, t_min, t_max, t):
        if t_min >= 0 and t_max <= 1:
            return t
        else:
            return (t - t_min) / (t_max - t_min)

    # def denormalize(self, t_min, t_max, t):
    #     return t * (t_max - t_min) + t_min
