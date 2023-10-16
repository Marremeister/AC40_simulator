import pandas as pd
from tqdm import tqdm
import os
#LÄgg till datum och namn i excel-fil

def best_overall_vmg_highlights(df):
    df = df.iloc[2100:]  # Starting from row 2100

    window_size = 210  # 7 seconds * 30Hz

    # Find best VMG window for a given condition
    def find_best_vmg_window(data):
        best_avg_vmg = float('-inf')
        best_start_index = None

        for start_idx in range(0, len(data) - window_size):
            window = data.iloc[start_idx:start_idx + window_size]

            avg_vmg = window['Boat.VMG_kts'].abs().mean()

            if avg_vmg > best_avg_vmg and \
                    ((window['Boat.FoilPort.Cant'] > 120) | (window['Boat.FoilStbd.Cant'] > 120)).any():
                best_avg_vmg = avg_vmg
                best_start_index = start_idx

        if best_start_index is not None:
            return data.iloc[best_start_index:best_start_index + window_size]
        return None

    # Filter data for upwind VMG (considering both positive and negative TWA ranges)
    upwind_filter = ((35 <= df['Boat.TWA']) & (df['Boat.TWA'] <= 50)) | (
                (-50 <= df['Boat.TWA']) & (df['Boat.TWA'] <= -35))
    upwind_data = df[upwind_filter]
    upwind_best_window = find_best_vmg_window(upwind_data)

    # Filter data for downwind VMG (considering both positive and negative TWA ranges)
    downwind_filter = ((130 <= df['Boat.TWA']) & (df['Boat.TWA'] <= 150)) | (
                (-150 <= df['Boat.TWA']) & (df['Boat.TWA'] <= -130))
    downwind_data = df[downwind_filter]
    downwind_best_window = find_best_vmg_window(downwind_data)

    return upwind_best_window, downwind_best_window



def read_data(filename):
    df = pd.read_excel(filename, engine='openpyxl', header=None)

    # Use 'Time' or any unique column name to identify the header
    header_row = df[df[0].str.contains('Time', na=False)].index[0]

    df.columns = df.iloc[header_row]
    df = df.drop(header_row)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def remove_tgt_columns(df):
    tgt_columns = [col for col in df.columns if "Tgt" in col]
    return df.drop(columns=tgt_columns)

def identify_legs(df):
    df['Leg'] = 0  # Initiate leg column with zeros
    current_leg = 1
    upwind = False
    upwind_count = 0
    downwind_count = 0

    for idx in tqdm(range(2100, len(df)), desc="Identifying legs"):
        twa = df.at[idx, 'Boat.TWA']
        if not upwind and (-90 <= twa <= 90):  # Adjusted for negative TWA
            upwind_count += 1
            downwind_count = 0  # Reset downwind count
            if upwind_count >= 1000:  # Adjusted to 300 for 10 seconds
                upwind = True
                current_leg += 1
                upwind_count = 0
        elif upwind and (twa < -90 or twa > 90):  # Adjusted for negative TWA
            downwind_count += 1
            upwind_count = 0  # Reset upwind count
            if downwind_count >= 1000:  # Adjusted to 300 for 10 seconds
                upwind = False
                current_leg += 1
                downwind_count = 0

        df.at[idx, 'Leg'] = current_leg
    print(current_leg)
    return df



def identify_gybes_and_tacks(df):
    maneuvers = []

    # Loop through the DataFrame and identify points around the change in TWA
    for idx in tqdm(range(165, len(df) - 165), desc="Identifying maneuvers"):
        time = df.at[idx, 'Time']
        twa = df.at[idx, 'Boat.TWA']

        maneuver_type = None
        # Identify tacks
        if -10 < twa < 10 and -10 < df.at[idx + 165, 'Boat.TWA'] < 10:
            if (df.at[idx - 1, 'Boat.TWA'] < 0 and twa > 0) or (df.at[idx - 1, 'Boat.TWA'] > 0 and twa < 0):
                maneuver_type = 'Tack'
        # Identify gybes
        elif (twa < -165 or twa > 165) and (df.at[idx + 165, 'Boat.TWA'] < -165 or df.at[idx + 165, 'Boat.TWA'] > 165):
            if (df.at[idx - 1, 'Boat.TWA'] < -165 and twa > 165) or (df.at[idx - 1, 'Boat.TWA'] > 165 and twa < -165):
                maneuver_type = 'Gybe'

        if maneuver_type:
            vmg_loss, start, end = compute_vmg_loss(df, idx)
            maneuver_data = df.iloc[start - 165:end + 330].copy()  # Extracting 10-second window around the maneuver

            avg_values = maneuver_data.mean(numeric_only=True)  # Calculate the average of numeric columns
            maneuver_data = pd.concat([maneuver_data, pd.DataFrame([avg_values])], ignore_index=True)
            maneuvers.append((time, maneuver_type, vmg_loss, maneuver_data))

    # Sort maneuvers by VMG loss
    sorted_maneuvers = sorted(maneuvers, key=lambda x: x[2])

    return [m for m in sorted_maneuvers if m[1] == 'Gybe'], [m for m in sorted_maneuvers if m[1] == 'Tack']


def best_vmg_highlights(df):
    # Group the data by the 'Leg' column
    grouped = df.groupby('Leg')

    best_vmg_sequences = []

    for name, group in tqdm(grouped, desc="Extracting best 5s VMG sequence from each leg"):
        best_avg_vmg = float("-inf")  # initialize to negative infinity
        best_sequence = None

        for i in range(0, len(group) - 150):  # slide a window of 150 rows for 5 seconds (given 30Hz data frequency)
            window = group.iloc[i:i + 150]

            valid_window = True  # A flag to track if current window is valid based on cant angles

            for idx, row in window.iterrows():
                # Check if both boards are below 120 degrees for this specific row
                if row['Boat.FoilPort.Cant'] <= 120 and row['Boat.FoilStbd.Cant'] <= 120:
                    valid_window = False
                    break

            # If the window is valid, calculate average VMG
            if valid_window:
                avg_vmg = abs(window['Boat.VMG_kts'].mean())

                if avg_vmg > best_avg_vmg:
                    best_avg_vmg = avg_vmg
                    best_sequence = window

        # Append the best sequence for this leg to the results list
        if best_sequence is not None:  # Ensure we only append valid sequences
            best_vmg_sequences.append(best_sequence)

    # Combine the results into a single DataFrame
    result_df = pd.concat(best_vmg_sequences, axis=0)

    # Append average values of the entire result dataframe to the bottom
    averages = result_df.mean(numeric_only=True).to_frame().T
    result_df = pd.concat([result_df, averages], ignore_index=True)

    return result_df


def compute_vmg_loss(df, idx):
    cant_port_column = 'Boat.FoilPort.Cant'
    cant_stbd_column = 'Boat.FoilStbd.Cant'

    is_canting = df.at[idx, cant_port_column] > 125 or df.at[idx, cant_stbd_column] > 125

    start_idx = idx
    while start_idx > 0 and is_canting:
        start_idx -= 1
        is_canting = df.at[start_idx, cant_port_column] > 125 or df.at[start_idx, cant_stbd_column] > 125

    if start_idx <= 0:
        return None, None, None

    avg_vmg_before = df.at[start_idx - 1, 'Boat.VMG_kts']

    end_idx = idx
    is_canting = df.at[end_idx, cant_port_column] < 125 or df.at[end_idx, cant_stbd_column] < 125
    while end_idx < len(df) - 1 and not is_canting:
        end_idx += 1
        is_canting = df.at[end_idx, cant_port_column] < 125 or df.at[end_idx, cant_stbd_column] < 125

    if end_idx >= len(df) - 1:
        return None, None, None

    avg_vmg_after = df.at[end_idx + 1, 'Boat.VMG_kts']

    vmg_loss = avg_vmg_before - avg_vmg_after

    return vmg_loss, start_idx, end_idx


def main():
    # Input the file paths
    source_file_path = input("Enter the path of the .xlsx file to analyze: ")
    if source_file_path == "":
        source_file_path = "Run_231014112244.xlsx"
    save_path = input(
        "Enter the path where the new file should be saved (without extension or press Enter for current directory): ")
    if not save_path.strip():  # if save_path is empty or whitespace
        save_path = os.path.join(os.getcwd(), "result")

    # User interaction for configuring output
    save_manoeuvres = input(
        "Do you want to save manoeuvres (gybes/tacks) to the Excel file? (yes/no): ").lower() == 'yes'
    save_vmg_highlights = input("Do you want VMG highlights in the Excel file? (yes/no): ").lower() == 'yes'

    vmg_highlight_choice = 'none'
    overall_data_or_avg = 'none'

    if save_vmg_highlights:
        vmg_highlight_choice = input(
            "Do you want VMG highlights for each leg or overall best VMG? (leg/overall): ").lower()
        overall_data_or_avg = input(
            "Do you want the entire data sequence or just the average for VMG? (overall/average): ").lower()

    # Load and preprocess the data
    df = read_data(source_file_path)
    df = remove_tgt_columns(df)
    df = identify_legs(df)

    # Construct a dynamic filename based on the extracted data
    filename_suffix = ''

    if save_manoeuvres:
        filename_suffix += "_Manoeuvres"
        gybes, tacks = identify_gybes_and_tacks(df)
        print(f"Identified {len(gybes)} gybes and {len(tacks)} tacks.")
    else:
        gybes, tacks = [], []

    vmg_highlights = None
    upwind_best, downwind_best = None, None
    if vmg_highlight_choice == 'leg':
        filename_suffix += "_LegVMG"
        vmg_highlights = best_vmg_highlights(df)
    elif vmg_highlight_choice == 'overall':
        filename_suffix += "_OverallVMG"
        upwind_best, downwind_best = best_overall_vmg_highlights(df)

    final_save_path = f"{save_path}{filename_suffix}.xlsx"

    # Save the results to Excel
    with pd.ExcelWriter(final_save_path) as writer:
        if save_manoeuvres:
            # Save Gybes
            for i, (time, _, _, maneuver_data) in enumerate(gybes, start=1):
                maneuver_data.to_excel(writer, sheet_name=f'Gybe{i}', index=False)

            # Save Tacks
            for i, (time, _, _, maneuver_data) in enumerate(tacks, start=1):
                maneuver_data.to_excel(writer, sheet_name=f'Tack{i}', index=False)

        if save_vmg_highlights and vmg_highlight_choice == 'leg':
            leg_numbers = vmg_highlights['Leg'].unique()
            for leg_number in leg_numbers:
                leg_data = vmg_highlights[vmg_highlights['Leg'] == leg_number]

                if overall_data_or_avg == 'average':
                    leg_data = pd.DataFrame(leg_data.mean()).transpose()

                leg_data.to_excel(writer, sheet_name=f'Leg_{leg_number}_VMG_Highlight', index=False)

        elif save_vmg_highlights and vmg_highlight_choice == 'overall':
            if overall_data_or_avg == 'average':
                upwind_best = pd.DataFrame(upwind_best.mean()).transpose()
                downwind_best = pd.DataFrame(downwind_best.mean()).transpose()

            upwind_best.to_excel(writer, sheet_name='Upwind_Overall_VMG_Highlight', index=False)
            downwind_best.to_excel(writer, sheet_name='Downwind_Overall_VMG_Highlight', index=False)

    print(f"Analysis complete. Results saved in '{final_save_path}'.")

if __name__ == "__main__":
    main()