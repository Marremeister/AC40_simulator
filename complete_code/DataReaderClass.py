import pandas as pd
import os


class DataAnalyzer:
    def __init__(self, filepath):
        self.df = self.read_data(filepath)

    @staticmethod
    def read_data(filename):
        # Determine file type and read accordingly
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, engine='openpyxl', header=None)
        else:
            raise ValueError("Unsupported file format")

        header_row = df[df[0].str.contains('Time', na=False)].index[0]
        df.columns = df.iloc[header_row]
        df = df.drop(header_row)
        df = df.reset_index(drop=True)

        # Convert specific columns to float (or int) as required
        columns_to_convert = ["Boat.VMG_kts", "Boat.Speed_kts", "Boat.TWA", "Boat.Heel",
                                               "Boat.Rudder.Angle", "Boat.Trim", "Boat.FoilPort.Cant", "Boat.FoilStbd.Cant", "Boat.Aero.MainTraveller"]  # Replace with actual column names
        for col in columns_to_convert:
            df[col] = df[col].astype(float)  # or use .astype(int) for integers

        return df

    def add_leg_col(self):
        self.df['tack'] = self.df['Boat.TWA'].apply(lambda x: 'starboard' if x > 0 else 'port')

    def remove_tgt_columns(self):
        tgt_columns = [col for col in self.df.columns if "Tgt" in col]
        self.df.drop(columns=tgt_columns, inplace=True)

    def convert_to_absolute(self):
        self.df['Boat.VMG_kts'] = self.df['Boat.VMG_kts'].abs()
        self.df['Boat.TWA'] = self.df['Boat.TWA'].abs()
