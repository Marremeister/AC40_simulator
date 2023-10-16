import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, filepath):
        self.df = self.read_data(filepath)

    @staticmethod
    def read_data(filename):
        df = pd.read_excel(filename, engine='openpyxl', header=None)
        header_row = df[df[0].str.contains('Time', na=False)].index[0]
        df.columns = df.iloc[header_row]
        df = df.drop(header_row)
        df = df.reset_index(drop=True)
        return df

    def remove_tgt_columns(self):
        tgt_columns = [col for col in self.df.columns if "Tgt" in col]
        self.df.drop(columns=tgt_columns, inplace=True)



class Maneuver:
    def __init__(self, dataframe):
        self.df = dataframe

    def identify_gybes_and_tacks(self):
        maneuvers = []

        collecting_data = False
        start_idx = None
        maneuver_type = None

        for idx in tqdm(range(1, len(self.df)), desc="Identifying maneuvers"):
            cant_port = self.df.at[idx, 'Boat.FoilPort.Cant']
            cant_stbd = self.df.at[idx, 'Boat.FoilStbd.Cant']
            prev_cant_port = self.df.at[idx - 1, 'Boat.FoilPort.Cant']
            prev_cant_stbd = self.df.at[idx - 1, 'Boat.FoilStbd.Cant']

            if not collecting_data and (prev_cant_port > 120 or prev_cant_stbd > 120) and \
                    (cant_port <= 120 or cant_stbd <= 120):
                collecting_data = True
                start_idx = idx

            if collecting_data:
                twa = self.df.at[idx, 'Boat.TWA']

                # Identify tacks
                if abs(twa) < 90:
                    if (self.df.at[idx - 1, 'Boat.TWA'] > 0 and twa < 0) or \
                            (self.df.at[idx - 1, 'Boat.TWA'] < 0 and twa > 0):
                        maneuver_type = 'Tack'

                # Identify gybes
                elif abs(twa) > 90:
                    if (self.df.at[idx - 1, 'Boat.TWA'] < -170 and twa > 170) or \
                            (self.df.at[idx - 1, 'Boat.TWA'] > 170 and twa < -170):
                        maneuver_type = 'Gybe'

                if (cant_port > 120 or cant_stbd > 120) and (prev_cant_port <= 120 or prev_cant_stbd <= 120):
                    collecting_data = False
                    if maneuver_type:  # if we found a maneuver during the data collection
                        vmg_loss, _, _ = self._compute_vmg_loss(start_idx, idx + 60)
                        maneuver_data = self.df.iloc[start_idx - 30:idx + 60].copy()
                        avg_values = maneuver_data.mean(numeric_only=True)
                        maneuver_data = pd.concat([maneuver_data, pd.DataFrame([avg_values])], ignore_index=True)
                        maneuvers.append((self.df.at[start_idx, 'Time'], maneuver_type, vmg_loss, maneuver_data))
                        maneuver_type = None  # reset for the next round

        # Sort maneuvers by VMG loss
        sorted_maneuvers = sorted(maneuvers, key=lambda x: x[2])

        return [m for m in sorted_maneuvers if m[1] == 'Gybe'], [m for m in sorted_maneuvers if m[1] == 'Tack']

    def _compute_vmg_loss(self, start_idx, end_idx):
        baseline_vmg = self.df.at[start_idx - 30, 'Boat.VMG_kts'] * 0.51444  # converting knots to m/s
        maneuver_data = self.df.iloc[start_idx:end_idx]['Boat.VMG_kts'].abs().values * 0.51444  # converting knots to m/s

        baseline_integral = baseline_vmg * (1 / 30) * len(maneuver_data)
        maneuver_integral = sum(maneuver_data) * (1 / 30)

        vmg_loss = baseline_integral - maneuver_integral

        return vmg_loss, start_idx, end_idx

    def save_maneuvers(self, output_filename='maneuvers.xlsx'):
        gybes, tacks = self.identify_gybes_and_tacks()

        if not gybes and not tacks:
            print("No maneuvers identified. Exiting without saving.")
            return

        with pd.ExcelWriter(output_filename) as writer:
            for i, (time, _, _, data) in enumerate(tacks, start=1):
                sheet_name = f"tack{i}"
                data.to_excel(writer, sheet_name=sheet_name, index=False)

            for i, (time, _, _, data) in enumerate(gybes, start=1):
                sheet_name = f"gybe{i}"
                data.to_excel(writer, sheet_name=sheet_name, index=False)

    def plot_meters_lost(self):
        gybes, tacks = self.identify_gybes_and_tacks()

        gybe_times = [gybe[0] for gybe in gybes]
        gybe_losses = [gybe[2] for gybe in gybes]  # Note: changed index from 3 to 2 to get the VMG loss

        tack_times = [tack[0] for tack in tacks]
        tack_losses = [tack[2] for tack in tacks]  # Note: changed index from 3 to 2 to get the VMG loss

        plt.scatter(gybe_times, gybe_losses, label='Gybes', color='blue')
        plt.scatter(tack_times, tack_losses, label='Tacks', color='red')
        plt.xlabel("Time of Maneuver Initiation (seconds)")
        plt.ylabel("Meters Lost")
        plt.legend()
        plt.title("Meters Lost per Maneuver Over Time")
        plt.show()

class VMG_Highlighter:
    def __init__(self, dataframe):
        self.df = dataframe

    def _is_valid_window(self, start_idx, twa_range, window_size=210):  # Adjusted window size to 210
        twa_values = self.df.iloc[start_idx:start_idx + window_size]['Boat.TWA'].abs().values
        cant_port_values = self.df.iloc[start_idx:start_idx + window_size]['Boat.FoilPort.Cant'].values
        cant_stbd_values = self.df.iloc[start_idx:start_idx + window_size]['Boat.FoilStbd.Cant'].values

        valid_twa = all(twa_range[0] <= twa <= twa_range[1] for twa in twa_values)
        valid_cant = all(cant_port > 120 or cant_stbd > 120 for cant_port, cant_stbd in zip(cant_port_values, cant_stbd_values))

        return valid_twa and valid_cant

    def _compute_best_vmg_for_window(self, twa_range):
        window_size = 210  # 7-second window at 30Hz
        best_start_idx = None
        best_avg_vmg = float('-inf')

        for idx in range(len(self.df) - window_size + 1):
            if self._is_valid_window(idx, twa_range, window_size):
                vmg_values = self.df.iloc[idx:idx + window_size]['Boat.VMG_kts'].abs().values
                avg_vmg = sum(vmg_values) / window_size

                if avg_vmg > best_avg_vmg:
                    best_avg_vmg = avg_vmg
                    best_start_idx = idx

        return self.df.iloc[best_start_idx:best_start_idx + window_size] if best_start_idx is not None else None

    def best_vmg_highlights(self):
        upwind_data = self._compute_best_vmg_for_window((35, 60))
        downwind_data = self._compute_best_vmg_for_window((125, 155))

        return upwind_data, downwind_data

    def best_overall_vmg_highlights(self):
        # Since the logic for computing the best VMG window is the same for both leg and overall,
        # we can use the previously defined function here as well
        return self.best_vmg_highlights()


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


