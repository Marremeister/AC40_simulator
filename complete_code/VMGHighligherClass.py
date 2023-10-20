from PlotterClass import Plotter
from tqdm import tqdm
import matplotlib.pyplot as plt

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


    def plot_leg_to_leg_vmg(self, segment1, segment2, leg, y_columns):
        pass

    def plot_vmg_segment(self, segment, title, y_columns):
        if not segment.empty:
            x_values = segment["Time"]

            # Convert TWA to absolute values
            if "Boat.TWA" in segment.columns:
                segment.loc[:, "Boat.TWA"] = segment["Boat.TWA"].abs()

            if "Boat.VMG_kts" in segment.columns:
                segment.loc[:, "Boat.VMG_kts"] = segment["Boat.VMG_kts"].abs()


            # Create the Plotter instance for individual subplots
            plotter_subplots = Plotter(x_values=x_values,
                                       title=title,
                                       xlabel="Time",
                                       ylabel="Value")

            # Plot the individual subplots
            rows, cols = Plotter.determine_subplot_dimensions(len(y_columns))

            # Create a figure and axes objects here for individual subplots
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))  # Adjust the figsize if necessary

            plotter_subplots.plot_subplots(y_columns, segment, title, rows, cols, fig, axs)

            self.plot_many_variables_together(segment, ["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA"])
            plt.show()

        else:
            print(f"No {title} found in VMG highlights.")

    def plot_many_variables_together(self, segment, col_combo, title = "Combo"):

        if not segment.empty:
            x_values = segment["Time"]

            if "Boat.TWA" in segment.columns:
                segment.loc[:, "Boat.TWA"] = segment["Boat.TWA"].abs()

            if "Boat.VMG_kts" in segment.columns:
                segment.loc[:, "Boat.VMG_kts"] = segment["Boat.VMG_kts"].abs()

            plotter_combined = Plotter(x_values=x_values,
                                       title=title + " - Combined",
                                       xlabel="Time",
                                       ylabel="Value")

            # Plot all columns together
            for column in col_combo:
                if column in segment.columns:
                    y_values = segment[column]
                    plotter_combined.add_line(y_values, label=column)

            plotter_combined.plot()

    def plot_vmg(self, segment, vmg=False, leg_highlights=[]):

        y_columns = ["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA", "Boat.Heel", "Boat.Aero.MainTraveller"]


        self.plot_vmg_segment(segment[0], "Upwind VMG-high", y_columns)
        self.plot_vmg_segment(segment[1], "Downwind VMG-high", y_columns)


        # if vmg:
        #     plot_differences = input("Do you want to plot the best and worst VMG-legs next to eachother? (yes/no) > ")
        #     if plot_differences == "yes":
        #         for i, tuple in enumerate(leg_highlights):
        #             if tuple[0] == "Upwind":
        #                 for j in range(i, len(leg_highlights)):
        #                     if leg_highlights[j][0] == "Upwind":
        #                         self.plot_leg_to_leg_vmg(tuple[1], leg_highlights[j][1], "Upwind", y_columns)
        #             if tuple[0] == "Downwind":
        #                 for j in range(i, len(leg_highlights)):
        #                     if leg_highlights[j][0] == "Downwind":
        #                         self.plot_leg_to_leg_vmg(tuple[1], leg_highlights[j][1], "Downwind", y_columns)
