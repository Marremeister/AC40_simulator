class LegIdentifier:
    def __init__(self, dataframe):
        self.df = dataframe
        self.legs = []  # A list to store the starting and ending indices of each leg

    def _check_section(self, start_idx, twa_range, section_length=300):  # 10-second section at 30Hz
        twa_values = self.df.iloc[start_idx:start_idx + section_length]['Boat.TWA'].abs().values
        return all(twa_range[0] <= twa <= twa_range[1] for twa in twa_values)

    def identify_legs(self):
        idx = 2100  # Starting index after the initial 2100 rows
        current_leg_start = idx
        current_direction = None

        while idx < len(self.df) - 300:  # Ensure there's always a 10-second section ahead to check
            if current_direction is None or current_direction == "downwind":
                # Check if it's an upwind leg
                if self._check_section(idx, (0, 90)):
                    if current_direction is not None:
                        self.legs.append((current_leg_start, idx))
                    current_leg_start = idx
                    current_direction = "upwind"

            if current_direction == "upwind":
                # Check if it's a downwind leg
                if self._check_section(idx, (90, 179)):
                    self.legs.append((current_leg_start, idx))
                    current_leg_start = idx
                    current_direction = "downwind"

            idx += 1

        # Add the last leg if the data ends before switching the leg
        if current_leg_start < len(self.df) - 1:
            self.legs.append((current_leg_start, len(self.df) - 1))

        return self.legs