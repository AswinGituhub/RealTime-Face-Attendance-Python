import os
import pandas as pd
from datetime import datetime

def mark_attendance_csv(name, file_path):
    """Marks attendance for a student in a CSV file."""
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)

    # Check if student already marked
    if name not in df['Name'].values:
        now = datetime.now().strftime("%H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(file_path, index=False)
        print(f"Marked attendance for {name}")
