from PlotterClass import Plotter
from tqdm import tqdm

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