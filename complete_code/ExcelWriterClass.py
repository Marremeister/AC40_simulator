import pandas as pd

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