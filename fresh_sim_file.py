import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, filepath):
        self.df = self._read_data(filepath)

    def _read_data(self, filename):
        df = pd.read_excel(filename, engine='openpyxl', header=None)
        header_row = df[df[0].str.contains('Time', na=False)].index[0]
        df.columns = df.iloc[header_row]
        return df.drop(header_row).reset_index(drop=True)

    def remove_tgt_columns(self):
        self.df.drop(columns=[col for col in self.df.columns if "Tgt" in col], inplace=True)


class Maneuver:
    def __init__(self, dataframe):
        self.df = dataframe

    def identify_gybes_and_tacks(self):
        maneuvers = []
        collecting_data = False
        start_idx = None
        maneuver_type = None

        for idx in tqdm(range(1, len(self.df)), desc="Identifying maneuvers"):
            cant_port, prev_cant_port = self.df.iloc[idx][['Boat.FoilPort.Cant', 'Boat.FoilStbd.Cant']]
            cant_stbd, prev_cant_stbd = self.df.iloc[idx - 1][['Boat.FoilPort.Cant', 'Boat.FoilStbd.Cant']]

            if not collecting_data and (prev_cant_port > 120 or prev_cant_stbd > 120) and \
                    (cant_port <= 120 or cant_stbd <= 120):
                collecting_data, start_idx = True, idx

            if collecting_data:
                twa, prev_twa = self.df.iloc[idx]['Boat.TWA'], self.df.iloc[idx - 1]['Boat.TWA']
                maneuver_type = self._get_maneuver_type(twa, prev_twa)

                if (cant_port > 120 or cant_stbd > 120) and (prev_cant_port <= 120 or prev_cant_stbd <= 120):
                    collecting_data = False
                    if maneuver_type:
                        vmg_loss = self._compute_vmg_loss(start_idx, idx + 60)[0]
                        maneuver_data = self.df.iloc[start_idx - 30:idx + 60].copy()
                        avg_values = maneuver_data.mean(numeric_only=True)
                        maneuver_data = pd.concat([maneuver_data, pd.DataFrame([avg_values])], ignore_index=True)
                        maneuvers.append((self.df.at[start_idx, 'Time'], maneuver_type, vmg_loss, maneuver_data))
                        maneuver_type = None

        return [m for m in sorted(maneuvers, key=lambda x: x[2]) if m[1] == 'Gybe'], \
               [m for m in sorted(maneuvers, key=lambda x: x[2]) if m[1] == 'Tack']

    def _get_maneuver_type(self, twa, prev_twa):
        if abs(twa) < 90 and (prev_twa > 0 > twa or prev_twa < 0 < twa):
            return 'Tack'
        elif abs(twa) > 90 and (prev_twa < -170 and twa > 170 or prev_twa > 170 and twa < -170):
            return 'Gybe'
        return None

    def _compute_vmg_loss(self, start_idx, end_idx):
        baseline_vmg = self.df.at[start_idx - 30, 'Boat.VMG_kts'] * 0.51444
        maneuver_data = self.df.iloc[start_idx:end_idx]['Boat.VMG_kts'].abs().values * 0.51444

        baseline_integral = baseline_vmg * len(maneuver_data) / 30
        maneuver_integral = sum(maneuver_data) / 30

        return baseline_integral - maneuver_integral, start_idx, end_idx

    def save_maneuvers(self, output_filename='maneuvers.xlsx'):
        gybes, tacks = self.identify_gybes_and_tacks()

        if not gybes and not tacks:
            print("No maneuvers identified. Exiting without saving.")
            return

        with pd.ExcelWriter(output_filename) as writer:
            for i, (_, _, _, data) in enumerate(tacks, start=1):
                data.to_excel(writer, sheet_name=f"tack{i}", index=False)
            for i, (_, _, _, data) in enumerate(gybes, start=1):
                data.to_excel(writer, sheet_name=f"gybe{i}", index=False)

    def plot_meters_lost(self):
        gybes, tacks = self.identify_gybes_and_tacks()

        plt.scatter([gybe[0] for gybe in gybes], [gybe[2] for gybe in gybes], c='b', marker='x', label='Gybes')
        plt.scatter([tack[0] for tack in tacks], [tack[2] for tack in tacks], c='r', marker='o', label='Tacks')
        plt.legend(loc="upper left")
        plt.title('Meters Lost in Maneuvers')
        plt.xlabel('Time')
        plt.ylabel('Meters Lost')
        plt.show()

class VMGHighlights:
    def __init__(self, dataframe):
        self.df = dataframe.iloc[2100:]  # Starting from row 2100

    def _find_best_vmg_window(self, data, window_size=210):
        best_avg_vmg = float('-inf')
        best_start_index = None

        for start_idx in range(0, len(data) - window_size):
            window = data.iloc[start_idx:start_idx + window_size]
            avg_vmg = window['Boat.VMG_kts'].abs().mean()

            if avg_vmg > best_avg_vmg and \
                    ((window['Boat.FoilPort.Cant'] > 120) | (window['Boat.FoilStbd.Cant'] > 120)).any():
                best_avg_vmg = avg_vmg
                best_start_index = start_idx

        if best_start_index is not None:
            return data.iloc[best_start_index:best_start_index + window_size]
        return None

    def best_overall_vmg_highlights(self):
        # Filter data for upwind VMG
        upwind_filter = ((35 <= self.df['Boat.TWA']) & (self.df['Boat.TWA'] <= 50)) | \
                        ((-50 <= self.df['Boat.TWA']) & (self.df['Boat.TWA'] <= -35))
        upwind_data = self.df[upwind_filter]
        upwind_best_window = self._find_best_vmg_window(upwind_data)

        # Filter data for downwind VMG
        downwind_filter = ((130 <= self.df['Boat.TWA']) & (self.df['Boat.TWA'] <= 150)) | \
                          ((-150 <= self.df['Boat.TWA']) & (self.df['Boat.TWA'] <= -130))
        downwind_data = self.df[downwind_filter]
        downwind_best_window = self._find_best_vmg_window(downwind_data)

        return upwind_best_window, downwind_best_window

    def best_vmg_highlights(self):
        grouped = self.df.groupby('Leg')
        best_vmg_sequences = []

        for name, group in tqdm(grouped, desc="Extracting best 5s VMG sequence from each leg"):
            best_avg_vmg = float("-inf")
            best_sequence = None

            for i in range(0, len(group) - 150):
                window = group.iloc[i:i + 150]
                valid_window = all(row['Boat.FoilPort.Cant'] > 120 or row['Boat.FoilStbd.Cant'] > 120 for _, row in window.iterrows())

                if valid_window:
                    avg_vmg = abs(window['Boat.VMG_kts'].mean())

                    if avg_vmg > best_avg_vmg:
                        best_avg_vmg = avg_vmg
                        best_sequence = window

            if best_sequence is not None:
                best_vmg_sequences.append(best_sequence)

        result_df = pd.concat(best_vmg_sequences, axis=0)
        averages = result_df.mean(numeric_only=True).to_frame().T
        result_df = pd.concat([result_df, averages], ignore_index=True)

        return result_df

    def save_highlights(self, upwind_sheet_name='upwind_best_vmg', downwind_sheet_name='downwind_best_vmg'):
        upwind_best, downwind_best = self.best_overall_vmg_highlights()
        with pd.ExcelWriter('vmg_highlights.xlsx') as writer:
            upwind_best.to_excel(writer, sheet_name=upwind_sheet_name, index=False)
            downwind_best.to_excel(writer, sheet_name=downwind_sheet_name, index=False)



if __name__ == '__main__':
    if os.path.exists('data.xlsx'):
        data_analyzer = DataAnalyzer('data.xlsx')
        data_analyzer.remove_tgt_columns()

        # Maneuver Analysis
        maneuver = Maneuver(data_analyzer.df)
        maneuver.save_maneuvers()
        maneuver.plot_meters_lost()

        # VMG Highlights
        vmg_highlights = VMGHighlights(data_analyzer.df)
        vmg_highlights.save_highlights()

        # Sail Analysis
        sail_analysis = SailAnalysis(data_analyzer.df)
        sail_analysis.save_analysis()
        sail_analysis.plot_analysis()

    else:
        print("File 'data.xlsx' not found!")