import pandas as pd
import os

def debug_ira_cols():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "excel", "IRA.xlsx")
    print(f"Reading {path}...")
    df = pd.read_excel(path)
    print("All Columns:")
    for c in df.columns:
        print(f"  {c}")
            
    # Print head of potential columns
    print("\nHead of first 5 columns:")
    print(df.iloc[:5, :5])

if __name__ == "__main__":
    debug_ira_cols()
