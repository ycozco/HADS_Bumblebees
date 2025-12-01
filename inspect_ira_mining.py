import pandas as pd
import os

files = [
    "IRA.xlsx",
    "COMPILADO METALES.xlsx",
    "COMPILADO NO METALES.xlsx"
]

for f in files:
    if os.path.exists(f):
        print(f"\n\n====== {f} ======")
        try:
            df = pd.read_excel(f, nrows=2)
            print("COLUMNS:", df.columns.tolist())
            print("HEAD:\n", df.head(1))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
