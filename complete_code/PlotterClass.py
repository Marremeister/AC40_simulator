import os
import matplotlib.pyplot as plt
#
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

    def plot(self, y_values=None, label=None):
        lines_to_plot = self.lines if y_values is None or y_values.empty else [(y_values, label)]

        for y, lbl in lines_to_plot:
            if self.plot_style == "scatter":
                if self.ax:
                    self.ax.scatter(self.x_values, y, label=lbl)
                else:
                    plt.scatter(self.x_values, y, label=lbl)
            elif self.plot_style == "line":
                if self.ax:
                    self.ax.plot(self.x_values, y, label=lbl)
                else:
                    plt.plot(self.x_values, y, label=lbl)
            else:
                raise ValueError(f"Unsupported plot style: {self.plot_style}")

        if self.ax:
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            if len(lines_to_plot) > 1:  # Only add a legend if there are multiple lines
                self.ax.legend()
        else:
            plt.title(self.title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            if len(lines_to_plot) > 1:  # Only add a legend if there are multiple lines
                plt.legend()

        if self.save_name:
            plt.savefig(os.path.join(self.save_path, self.save_name))

        self.lines = []



    def set_plot_style(self, style):
        self.plot_style = style


    def plot_subplots(self, y_columns, maneuver_df, maneuver_type, rows, cols, fig, axs):
        """
        Plots subplots based on y_columns provided.

        Args:
        - y_columns (list): columns to be plotted along y-axis
        - maneuver_df (DataFrame): data for the maneuver
        - maneuver_type (str): either 'tack' or 'gybe'
        - rows (int): number of subplot rows
        - cols (int): number of subplot columns
        - fig (matplotlib figure): main figure
        - axs (matplotlib axes): individual axes for subplots
        """

        for idx, column in enumerate(y_columns):
            # Handle individual columns
            if column in maneuver_df.columns:
                ax_idx = idx
                i, j = divmod(ax_idx, cols)
                ax = axs[j] if (rows == 1 or cols == 1) else axs[i, j]

                # Updated the way we call the plot method
                self.title = f"{maneuver_type.capitalize()} - {column}"
                self.ax = ax
                self.plot(y_values=maneuver_df[column], label=column)

    def plot_combined(self, y_columns, maneuver_df, title, xlabel="Time", ax=None):
        """
        Plot multiple columns on a single graph.

        Args:
        - y_columns (list): List of columns to plot.
        - maneuver_df (pd.DataFrame): DataFrame containing the data.
        - title (str): Plot title.
        - xlabel (str): x-axis label.
        - ax (matplotlib.Axes): Optional, the axis to plot on. If None, a new figure and axis are created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        for column in y_columns:
            if column in maneuver_df.columns:
                y_values = maneuver_df[column]
                self.add_line(y_values, label=column)

        self.title = title
        self.xlabel = xlabel
        self.ax = ax
        self.plot()

    @staticmethod
    def determine_subplot_dimensions(n):
        cols = 3
        rows = n // cols + (1 if n % cols else 0)

        # Adjust for the case when there are less than 4 columns
        if n <= 3:
            rows, cols = 1, n

        return rows, cols