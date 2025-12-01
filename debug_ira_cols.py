import pandas as pd

def debug_ira_cols():
    print("Reading IRA.xlsx...")
    df = pd.read_excel("IRA.xlsx")
    print("All Columns:")
    for c in df.columns:
        print(f"  {c}")
            
    # Print head of potential columns
    print("\nHead of first 5 columns:")
    print(df.iloc[:5, :5])

if __name__ == "__main__":
    debug_ira_cols()
