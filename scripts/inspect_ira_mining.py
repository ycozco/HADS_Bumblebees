import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_EXCEL = os.path.join(BASE_DIR, "data", "excel")

files = [
    os.path.join(DATA_EXCEL, "IRA.xlsx"),
    os.path.join(DATA_EXCEL, "COMPILADO METALES.xlsx"),
    os.path.join(DATA_EXCEL, "COMPILADO NO METALES.xlsx")
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
