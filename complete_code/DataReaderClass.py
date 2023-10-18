import pandas as pd

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

    def convert_to_absolute(self):
        self.df['Boat.VMG_kts'] = self.df['Boat.VMG_kts'].abs()
        self.df['Boat.TWA'] = self.df['Boat.TWA'].abs()