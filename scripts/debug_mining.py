import pandas as pd
import unidecode
import os

def clean_distrito(name):
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).upper().strip()
    return unidecode.unidecode(name)

def process_mining(metals_path, non_metals_path):
    print("Processing Mining data...")
    
    def process_one_mining_file(path, type_label):
        print(f"--- Processing {path} ---")
        try:
            # Try header=1
            df = pd.read_excel(path, header=1)
            print(f"Read with header=1. Columns: {df.columns.tolist()[:10]}...")
            
            # Check if empty or wrong header
            if 'ESTRATO' not in [str(c).upper().strip() for c in df.columns]:
                print("ESTRATO not found with header=1. Trying header=0...")
                df = pd.read_excel(path, header=0)
                print(f"Read with header=0. Columns: {df.columns.tolist()[:10]}...")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return pd.DataFrame()
            
        df.columns = [str(c).upper().strip() for c in df.columns]
        
        months = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']
        available_months = [m for m in months if m in df.columns]
        print(f"Available months: {available_months}")
        
        dist_col = None
        for c in df.columns:
            if 'DISTRITO' in c:
                dist_col = c
                break
        print(f"District column found: {dist_col}")
        
        if not dist_col:
            print("Error: District column not found")
            return pd.DataFrame()

        id_vars = ['ESTRATO', 'AÑO', dist_col]
        id_vars = [c for c in id_vars if c in df.columns]
        print(f"ID vars: {id_vars}")
        
        if not available_months:
            print("Error: No month columns found")
            return pd.DataFrame()
            
        df_melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=available_months,
            var_name='MES_STR',
            value_name='PRODUCCION'
        )
        
        print(f"Melted shape: {df_melted.shape}")
        return df_melted

    df_metals = process_one_mining_file(metals_path, "METALES")
    df_non_metals = process_one_mining_file(non_metals_path, "NO_METALES")
    
    df_combined = pd.concat([df_metals, df_non_metals], ignore_index=True)
    print(f"Combined shape: {df_combined.shape}")
    return df_combined

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_EXCEL = os.path.join(BASE_DIR, "data", "excel")
    process_mining(
        os.path.join(DATA_EXCEL, "COMPILADO METALES.xlsx"),
        os.path.join(DATA_EXCEL, "COMPILADO NO METALES.xlsx")
    )
