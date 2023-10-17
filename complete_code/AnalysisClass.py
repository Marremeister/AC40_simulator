from ExcelWriterClass import ExcelWriter
from LegIdentifierClass import LegIdentifier
from ManeuverClass import Maneuver
from VMGHighligherClass import VMG_Highlighter
import os
import pandas as pd
from datetime import datetime


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
