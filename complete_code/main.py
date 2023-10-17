from DataReaderClass import DataAnalyzer
from AnalysisClass import Analysis


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