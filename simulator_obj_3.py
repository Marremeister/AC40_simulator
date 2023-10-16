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

    def best_vmg_highlights(self):
        grouped = self.df.groupby('Leg')

        best_vmg_sequences = []

        for name, group in tqdm(grouped, desc="Extracting best 5s VMG sequence from each leg"):
            best_avg_vmg = float("-inf")
            best_sequence = None

            for i in range(0, len(group) - 150):
                window = group.iloc[i:i + 150]

                valid_window = True

                for idx, row in window.iterrows():
                    if row['Boat.FoilPort.Cant'] <= 120 and row['Boat.FoilStbd.Cant'] <= 120:
                        valid_window = False
                        break

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

    def best_overall_vmg_highlights(self):
        df = self.df.iloc[2100:]
        window_size = 210

        def find_best_vmg_window(data):
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

        upwind_filter = ((35 <= df['Boat.TWA']) & (df['Boat.TWA'] <= 50)) | (
                (-50 <= df['Boat.TWA']) & (df['Boat.TWA'] <= -35))
        upwind_data = df[upwind_filter]
        upwind_best_window = find_best_vmg_window(upwind_data)

        downwind_filter = ((130 <= df['Boat.TWA']) & (df['Boat.TWA'] <= 150)) | (
                (-150 <= df['Boat.TWA']) & (df['Boat.TWA'] <= -130))
        downwind_data = df[downwind_filter]
        downwind_best_window = find_best_vmg_window(downwind_data)

        return upwind_best_window, downwind_best_window

class SailAnalysis:

    def __init__(self):
        self.source_file_path = self.get_input("Enter the path of the .xlsx file to analyze: ", default="Run_231014112244.xlsx")
        self.save_path = self.get_input("Enter the path where the new file should be saved (without extension or press Enter for current directory): ", default=os.path.join(os.getcwd(), "result"))
        self.save_manoeuvres = self.get_input("Do you want to save manoeuvres (gybes/tacks) to the Excel file? (yes/no): ").lower() == 'yes'
        self.save_vmg_highlights = self.get_input("Do you want VMG highlights in the Excel file? (yes/no): ").lower() == 'yes'
        self.vmg_highlight_choice = self.get_input("Do you want VMG highlights for each leg or overall best VMG? (leg/overall): ") if self.save_vmg_highlights else 'none'
        self.overall_data_or_avg = self.input_overall_or_average()

        self.data_analyzer = DataAnalyzer(self.source_file_path)
        self.maneuver = Maneuver(self.data_analyzer.df)
        self.vmg_highlighter = VMG_Highlighter(self.data_analyzer.df)

    @staticmethod
    def get_input(prompt, default=None):
        response = input(prompt)
        return response if response else default

    def input_overall_or_average(self):
        choice = ""
        while choice not in ['overall', 'average']:
            choice = input("Do you want to get overall data or average? (overall/average): ").strip().lower()
        self.overall_data_or_avg = choice
        print(f"Selected: {self.overall_data_or_avg}")

    def run_analysis(self):
        self.data_analyzer.remove_tgt_columns()

        filename_suffix = ''
        gybes, tacks = [], []
        if self.save_manoeuvres:
            filename_suffix += "_Manoeuvres"

            # THIS IS WHERE YOU WOULD CALL save_maneuvers
            m = Maneuver(self.data_analyzer.df)
            m.save_maneuvers()
            # END OF INSERTED CODE

            gybes, tacks = self.maneuver.identify_gybes_and_tacks()
            print(f"Identified {len(gybes)} gybes and {len(tacks)} tacks.")

        vmg_highlights = None
        upwind_best, downwind_best = None, None
        if self.vmg_highlight_choice == 'leg':
            filename_suffix += "_LegVMG"
            vmg_highlights = self.vmg_highlighter.best_vmg_highlights()
        elif self.vmg_highlight_choice == 'overall':
            filename_suffix += "_OverallVMG"
            upwind_best, downwind_best = self.vmg_highlighter.best_overall_vmg_highlights()

        final_save_path = f"{self.save_path}{filename_suffix}.xlsx"
        self.save_to_excel(final_save_path, gybes, tacks, vmg_highlights, upwind_best, downwind_best)

        print(f"Analysis complete. Results saved in '{final_save_path}'.")

        # Here, plot the graph
        if self.save_manoeuvres:  # Only plot if the maneuvers are being saved
            self.maneuver.plot_meters_lost()

    def save_to_excel(self, path, gybes, tacks, vmg_highlights, upwind_best, downwind_best):
        with pd.ExcelWriter(path) as writer:
            # Save gybes
            if gybes:
                for i, (time, _, _, maneuver_data) in enumerate(gybes, start=1):
                    maneuver_data.to_excel(writer, sheet_name=f'Gybe{i}', index=False)

            # Save tacks
            if tacks:
                for i, (time, _, _, maneuver_data) in enumerate(tacks, start=1):
                    maneuver_data.to_excel(writer, sheet_name=f'Tack{i}', index=False)

            # Save VMG highlights
            if vmg_highlights is not None:
                leg_numbers = vmg_highlights['Leg'].unique()
                for leg_number in leg_numbers:
                    leg_data = vmg_highlights[vmg_highlights['Leg'] == leg_number]

                    if self.overall_data_or_avg == 'average':
                        leg_data = pd.DataFrame(leg_data.mean()).transpose()

                    leg_data.to_excel(writer, sheet_name=f'Leg_{leg_number}_VMG_Highlight', index=False)

            # Save overall best VMG
            if upwind_best is not None:
                if self.overall_data_or_avg == "overall":
                    upwind_best.to_excel(writer, sheet_name='Upwind_Overall_VMG_Highlight', index=False)
                else:  # save just the averages (last row)
                    avg_values = upwind_best.tail(1)
                    avg_values.to_excel(writer, sheet_name='Upwind_Overall_VMG_Highlight', index=False)

            if downwind_best is not None:
                if self.overall_data_or_avg == "overall":
                    downwind_best.to_excel(writer, sheet_name='Downwind_Overall_VMG_Highlight', index=False)
                else:  # save just the averages (last row)
                    avg_values = downwind_best.tail(1)
                    avg_values.to_excel(writer, sheet_name='Downwind_Overall_VMG_Highlight', index=False)


if __name__ == "__main__":
    analysis = SailAnalysis()
    analysis.run_analysis()
