import pandas as pd
import os

files = [
    "POBLACION.xlsx",
    "IRA.xlsx",
    "COMPILADO METALES.xlsx",
    "COMPILADO NO METALES.xlsx"
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
