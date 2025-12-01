import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_EXCEL = os.path.join(BASE_DIR, "data", "excel")

files = [
    os.path.join(DATA_EXCEL, "POBLACION.xlsx"),
    os.path.join(DATA_EXCEL, "IRA.xlsx"),
    os.path.join(DATA_EXCEL, "COMPILADO METALES.xlsx"),
    os.path.join(DATA_EXCEL, "COMPILADO NO METALES.xlsx")
]

for f in files:
    if os.path.exists(f):
        print(f"--- {f} ---")
        try:
            df = pd.read_excel(f, nrows=5)
            print(df.columns.tolist())
            print(df.head(2))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
    print("\n")
