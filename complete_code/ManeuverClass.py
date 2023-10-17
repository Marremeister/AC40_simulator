from PlotterClass import Plotter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


#TODO - VMG och boatspeed
#TODO - Enskilda grafer på roder, heel, travare
#TODO - heel - travare samma

class Maneuver:
    def __init__(self, dataframe):
        self.df = dataframe
        self.tacks = []
        self.gybes = []
        self._identify_gybes_and_tacks()

    def _identify_gybes_and_tacks(self):
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

        # Store maneuvers in self.tacks and self.gybes
        self.gybes = [m for m in maneuvers if m[1] == 'Gybe']
        self.tacks = [m for m in maneuvers if m[1] == 'Tack']

    def _compute_vmg_loss(self, start_idx, end_idx):
        baseline_vmg = self.df.at[start_idx - 30, 'Boat.VMG_kts'] * 0.51444  # converting knots to m/s
        maneuver_data = self.df.iloc[start_idx:end_idx]['Boat.VMG_kts'].abs().values * 0.51444  # converting knots to m/s

        baseline_integral = baseline_vmg * (1 / 30) * len(maneuver_data)
        maneuver_integral = sum(maneuver_data) * (1 / 30)

        vmg_loss = baseline_integral - maneuver_integral

        return vmg_loss, start_idx, end_idx

    def save_maneuvers(self, output_filename='maneuvers.xlsx'):
        if not self.gybes and not self.tacks:
            print("No maneuvers identified. Exiting without saving.")
            return

        with pd.ExcelWriter(output_filename) as writer:
            for i, (time, _, _, data) in enumerate(self.tacks, start=1):
                sheet_name = f"tack{i}"
                data.to_excel(writer, sheet_name=sheet_name, index=False)

            for i, (time, _, _, data) in enumerate(self.gybes, start=1):
                sheet_name = f"gybe{i}"
                data.to_excel(writer, sheet_name=sheet_name, index=False)

    def plot_meters_lost(self):
        gybe_times = [gybe[0] for gybe in self.gybes]
        gybe_losses = [gybe[2] for gybe in self.gybes]
        tack_times = [tack[0] for tack in self.tacks]
        tack_losses = [tack[2] for tack in self.tacks]

        # Initialize the Plotter class for Gybes
        gybe_plotter = Plotter(x_values=gybe_times,
                               title="Meters Lost per Gybe Over Time",
                               xlabel="Time of Gybe Initiation (seconds)",
                               ylabel="Meters Lost",
                               plot_style="scatter")
        gybe_plotter.add_line(gybe_losses, label="Gybe Loss")
        gybe_plotter.plot()

        # Initialize the Plotter class for Tacks
        tack_plotter = Plotter(x_values=tack_times,
                               title="Meters Lost per Tack Over Time",
                               xlabel="Time of Tack Initiation (seconds)",
                               ylabel="Meters Lost",
                               plot_style="scatter")
        tack_plotter.add_line(tack_losses, label="Tack Loss")
        tack_plotter.plot()

    def plot_best_maneuver(self, maneuver_type,
                           y_columns=["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA", "Boat.Heel", "Boat.FoilPort.Cant",
                                      "Boat.FoilStbd.Cant", "Boat.Rudder.Angle"]):
        """
        Plots the best maneuver (tack or gybe) based on minimum VMG loss.

        Args:
        - maneuver_type (str): either 'tack' or 'gybe'
        - y_columns (list): list of columns to be plotted along y-axis
        """

        # Select the maneuver list based on the type
        maneuvers = self.gybes if maneuver_type == "gybe" else self.tacks

        # Determine the best maneuver based on VMG loss
        best_maneuver = None
        best_value = float('inf')

        for maneuver in maneuvers:
            current_value = maneuver[2]
            if current_value < best_value:
                best_value = current_value
                best_maneuver = maneuver

        if best_maneuver:
            start_time, _, _, df = best_maneuver

            # Get the row corresponding to the start_time
            start_row = self.df[self.df['Time'] == start_time].index[0]

            # Get the end row based on the length of the df for the maneuver
            end_row = start_row + len(df)
            maneuver_df = self.df.iloc[start_row:end_row].copy()

            # Convert VMG to absolute value
            if "Boat.VMG_kts" in maneuver_df.columns:
                maneuver_df["Boat.VMG_kts"] = maneuver_df["Boat.VMG_kts"].abs()

            combined_columns = [["Boat.VMG_kts", "Boat.Speed_kts"], ["Boat.Heel", "Boat.Aero.MainTraveller"]]

            # Use the Plotter to plot everything in one subplot setup
            plotter_instance = Plotter(maneuver_df["Time"])

            # Determine subplot dimensions and create fig and axs objects
            n = len(y_columns)  # Number of plots
            rows, cols = Plotter.determine_subplot_dimensions(n)
            fig, axs = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figsize as necessary

            plotter_instance.plot_subplots(y_columns, maneuver_df, maneuver_type, rows, cols, fig, axs)

            for column in combined_columns:
                plotter_instance.plot_combined(column, maneuver_df, f"{maneuver_type.capitalize()} - {' & '.join(column)}")

        else:
            print(f"No best {maneuver_type} found in maneuvers.")
