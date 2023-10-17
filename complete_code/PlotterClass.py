import os
import matplotlib.pyplot as plt

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

    def plot_subplots(self, y_columns, maneuver_df, maneuver_type, rows=1, cols=3, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust the figsize to your liking

        for idx, column in enumerate(y_columns):
            i, j = divmod(idx, cols)
            if column in maneuver_df.columns:
                y_values = maneuver_df[column]
                single_column_plotter = Plotter(x_values=self.x_values,
                                                title=f"{maneuver_type.capitalize()} - {column}",
                                                xlabel=self.xlabel,
                                                ylabel=self.ylabel,
                                                ax=axs[j] if (rows == 1 or cols == 1) else axs[i, j])
                single_column_plotter.add_line(y_values, label=column)
                single_column_plotter.plot()

        plt.tight_layout()
        plt.show()

    def plot_combined(self, y_columns, maneuver_df, title, xlabel="Time"):
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