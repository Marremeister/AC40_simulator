import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


#TODO - Fixa plottern med ChatGPT
#TODO - Bryt ner analysis-klassen i fler metoder.
#TODO - Fråga en input om vad man vill se plottas till manövern. Detsamma kan göras med bästa VMG-sekvens.


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

    def plot_best_maneuver(self, maneuver_type, y_columns=["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA", "Boat.Heel", "Boat.FoilPort.Cant", "Boat.FoilStbd.Cant"]):
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

            if "Boat.TWA" in maneuver_df.columns:
                maneuver_df["Boat.TWA"] = maneuver_df["Boat.TWA"].abs()

            x_values = maneuver_df["Time"]

            # Create the Plotter instance
            plotter = Plotter(x_values=x_values,
                              title=f"{maneuver_type.capitalize()} Maneuver",
                              xlabel="Time",
                              ylabel="Value")

            # Plot the individual subplots
            rows, cols = Plotter.determine_subplot_dimensions(len(y_columns))
            plotter.plot_subplots(y_columns, maneuver_df, maneuver_type, rows, cols)

            # Plot all columns together
            for column in y_columns:
                if column in maneuver_df.columns:
                    y_values = maneuver_df[column]
                    plotter.add_line(y_values, label=column)

            plotter.plot()

        else:
            print(f"No best {maneuver_type} found in maneuvers.")


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

        for idx in tqdm(range(len(self.df) - window_size + 1), desc="Computing Best VMG"):
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

    def plot_vmg_segment(self, segment, title, y_columns=["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA", "Boat.Heel"]):
        """
        Plots a given VMG segment.

        Args:
        - segment: the VMG segment to plot
        - title (str): title for the plot
        - y_columns (list): list of columns to be plotted along y-axis
        """

        if not segment.empty:
            x_values = segment["Time"]

            # Convert TWA to absolute values
            if "Boat.TWA" in segment.columns:
                segment.loc[:, "Boat.TWA"] = segment["Boat.TWA"].abs()

            # Create the Plotter instance
            plotter = Plotter(x_values=x_values,
                              title=title,
                              xlabel="Time",
                              ylabel="Value")

            # Plot the individual subplots
            rows, cols = Plotter.determine_subplot_dimensions(len(y_columns))
            plotter.plot_subplots(y_columns, segment, title, rows, cols)

            # Plot all columns together
            for column in y_columns:
                if column in segment.columns:
                    y_values = segment[column]
                    plotter.add_line(y_values, label=column)

            plotter.plot()

        else:
            print(f"No {title} found in VMG highlights.")


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


class Plotter:
    def __init__(self, x_values, title="", xlabel="", ylabel="", plot_style="line", ax=None):
        self.x_values = x_values
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_style = plot_style  # Determines the style of the plot, default is 'line'
        self.ax = ax
        self.save_name = ""
        self.save_path = os.getcwd()  # Default to current directory
        self.lines = []  # Store multiple lines for plotting


    def set_save_name(self, name):
        self.save_name = name


    def set_save_path(self, path):
        self.save_path = path


    def add_line(self, y_values, label=None):
        self.lines.append((y_values, label))


    def plot(self):
        for y_values, label in self.lines:
            if self.plot_style == "scatter":
                if self.ax:
                    self.ax.scatter(self.x_values, y_values, label=label)
                else:
                    plt.scatter(self.x_values, y_values, label=label)
            elif self.plot_style == "line":
                plt.grid = True
                if self.ax:
                    self.ax.plot(self.x_values, y_values, label=label)
                else:
                    plt.plot(self.x_values, y_values, label=label)
            else:
                raise ValueError(f"Unsupported plot style: {self.plot_style}")

        if self.ax:
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            if len(self.lines) > 1:  # Only add a legend if there are multiple lines
                self.ax.legend()
        else:
            plt.title(self.title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            if len(self.lines) > 1:  # Only add a legend if there are multiple lines
                plt.legend()

        if self.save_name:
            plt.savefig(os.path.join(self.save_path, self.save_name))
        elif not self.ax:
            plt.show()


    def set_plot_style(self, style):
        self.plot_style = style


    def plot_subplots(self, y_columns, maneuver_df, maneuver_type, rows=1, cols=3):
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust the figsize to your liking

        for idx, column in enumerate(y_columns):
            i, j = divmod(idx, cols)
            if column in maneuver_df.columns:
                y_values = maneuver_df[column]
                single_column_plotter = Plotter(x_values=self.x_values,
                                                title=f"{maneuver_type.capitalize()} - {column}",
                                                xlabel=self.xlabel,
                                                ylabel=self.ylabel,
                                                ax=axs[i, j] if rows > 1 else axs[j])
                single_column_plotter.add_line(y_values, label=column)
                single_column_plotter.plot()

        plt.tight_layout()
        plt.show()


    @staticmethod
    def determine_subplot_dimensions(n):
        cols = 3
        rows = n // cols + (1 if n % cols else 0)

        # Adjust for the case when there are less than 4 columns
        if n <= 3:
            rows, cols = 1, n

        return rows, cols


class ExcelWriter:
    def __init__(self, filename):
        self.filename = filename
        self.writer = None
        self.book = None

    def set_filename(self, filename):
        self.filename = filename

    def write_data(self, data, sheet_name):
        if not isinstance(self.writer, pd.ExcelWriter):
            # Initialize the ExcelWriter instance
            self.writer = pd.ExcelWriter(self.filename, engine='openpyxl')
            # You don't need to load an existing book since you're creating a new one.
        # Now, write the data to the specified sheet name
        print(f"Writing data to sheet: {sheet_name}")
        data.to_excel(self.writer, sheet_name=sheet_name, index=False)

    def save(self):
        # Check for at least one visible sheet
        visible_sheets = [sheet for sheet in self.writer.book if not sheet.sheet_state == 'hidden']
        if not visible_sheets:
            # Make the first sheet visible if none are
            self.writer.book.active = self.writer.book.worksheets[0]
            self.writer.book.active.sheet_state = 'visible'

        # Set an active sheet
        self.writer.book.active = self.writer.book.worksheets[0]

        # Save the Excel file
        self.writer.close()


class Analysis:
    def __init__(self, dataframe):
        self.df = dataframe
        self.legs = []
        self.excel_writer = None
        self.output_path = ''
        self.filename = ''
        self.analyze_maneuvers = ''
        self.highlight_vmg = ''

    def _identify_legs(self):
        leg_identifier = LegIdentifier(self.df)
        self.legs = leg_identifier.identify_legs()

    def _gather_inputs(self):
        self.output_path = self._get_output_path()
        self.filename, self.analyze_maneuvers, self.highlight_vmg = self._setup_filename()

    def _get_output_path(self):
        path = input("Enter path to save output file (Press Enter for current directory): ")
        return path if path else os.getcwd()

    def _setup_filename(self):
        today_date = datetime.today().strftime('%Y%m%d')
        filename_parts = [today_date]

        self.analyze_maneuvers = input("Do you want to analyze maneuvers? (yes/no): ").lower()
        if self.analyze_maneuvers == "yes":
            filename_parts.append("Maneuver")

        self.highlight_vmg = input("Do you want to highlight VMG? (yes/no): ").lower()
        if self.highlight_vmg == "yes":
            filename_parts.append("VMG")

        return "_".join(filename_parts) + ".xlsx", self.analyze_maneuvers, self.highlight_vmg

    def _init_excel_writer(self):
        self.excel_writer = ExcelWriter(os.path.join(self.output_path, self.filename))

    def _analyze_maneuvers(self):
        maneuvers = Maneuver(self.df)
        gybes, tacks = maneuvers.gybes, maneuvers.tacks

        # Sort maneuvers by VMG loss
        gybes = sorted(gybes, key=lambda x: x[2])
        tacks = sorted(tacks, key=lambda x: x[2])

        for i, (_, _, _, data) in enumerate(gybes, start=1):
            self.excel_writer.write_data(data, f"Gybe{i}")
        for i, (_, _, _, data) in enumerate(tacks, start=1):
            self.excel_writer.write_data(data, f"Tack{i}")

        plot_best_maneuver = input("Should we plot the best tack and gybe? (no, both, gybe or tack) ").lower()
        if plot_best_maneuver == "both":
            maneuvers.plot_best_maneuver("tack")
            maneuvers.plot_best_maneuver("gybe")
        elif plot_best_maneuver == "tack":
            maneuvers.plot_best_maneuver("tack")
        elif plot_best_maneuver == "gybe":
            maneuvers.plot_best_maneuver("gybe")

        maneuvers.plot_meters_lost()

    def _highlight_vmg(self):
        vmg_highlighter = VMG_Highlighter(self.df)
        avg_or_whole = input(
            "Do you want to output the average of each column or the whole data sequence? (average/whole): ").lower()

        # Calculate overall best VMG and save it
        overall_highlights = vmg_highlighter.best_overall_vmg_highlights()
        for idx, data in [('Upwind', overall_highlights[0]), ('Downwind', overall_highlights[1])]:
            if avg_or_whole == 'average' and data is not None:
                data = pd.DataFrame(data.mean()).transpose()
            if data is not None:
                self.excel_writer.write_data(data, f"OverallVMGHighlight{idx}")

        plot_vmg_high = input("Do you want to plot parameters from the VMG highlight answer: no, upwind, downwind, both ")

        if plot_vmg_high == "upwind":
            vmg_highlighter.plot_vmg_segment(overall_highlights[0], "Upwind VMG-high")
        elif plot_vmg_high == "downwind:":
            vmg_highlighter.plot_vmg_segment(overall_highlights[1], "Downwind VMG-high")
        elif plot_vmg_high == "both":
            vmg_highlighter.plot_vmg_segment(overall_highlights[0], "Upwind VMG-high")
            vmg_highlighter.plot_vmg_segment(overall_highlights[1], "Downwind VMG-high")


        # If user wants VMG highlights for each leg, then calculate and save those as well
        per_leg_or_overall = input(
            "Do you want to save VMG highlights for each leg or overall? (leg/overall): ").lower()

        if per_leg_or_overall == 'leg':
            self._identify_legs()
            leg_highlights = []

            for start, end in self.legs:
                leg_df = self.df.iloc[start:end]
                leg_highlighter = VMG_Highlighter(leg_df)
                upwind_data, downwind_data = leg_highlighter.best_vmg_highlights()

                # Process data based on user choice and store
                if upwind_data is not None:
                    if avg_or_whole == 'average':
                        upwind_data = pd.DataFrame(upwind_data.mean()).transpose()
                    leg_highlights.append(('Upwind', upwind_data))

                if downwind_data is not None:
                    if avg_or_whole == 'average':
                        downwind_data = pd.DataFrame(downwind_data.mean()).transpose()
                    leg_highlights.append(('Downwind', downwind_data))

            # Write each entry in leg_highlights to Excel
            for idx, (direction, data) in enumerate(leg_highlights, start=1):
                self.excel_writer.write_data(data, f"LegVMGHighlight{direction}{idx}")

    def _save_data(self):
        self.excel_writer.save()
        print("Data has been saved successfully!")

    def start(self):
        self._gather_inputs()
        self._init_excel_writer()

        if self.analyze_maneuvers == "yes":
            self._analyze_maneuvers()

        if self.highlight_vmg == "yes":
            self._highlight_vmg()

        self._save_data()



if __name__ == "__main__":
    input_path = input("Enter path to file to analyze: ")
    if not input_path:
        input_path = "Run_231014112244.xlsx"

    # Initialize the DataAnalyzer
    data_analyzer = DataAnalyzer(input_path)

    # Remove the Tgt columns (if necessary)
    data_analyzer.remove_tgt_columns()

    # Step 2: Now use the processed dataframe to initialize the Analysis class and start the analysis
    analysis = Analysis(data_analyzer.df)
    analysis.start()
