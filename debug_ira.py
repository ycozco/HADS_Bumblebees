import pandas as pd
import unidecode
import os

def clean_distrito(name):
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).upper().strip()
    return unidecode.unidecode(name)

def process_ira(file_path):
    print(f"Processing IRA data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading IRA file: {e}")
        return pd.DataFrame()
    
    # Standardize columns: lowercase and strip
    df.columns = [str(c).lower().strip() for c in df.columns]
    print(f"IRA Columns: {df.columns.tolist()}")
    
    # Check required columns
    required = ['distrito', 'semana']
    for req in required:
        if req not in df.columns:
            print(f"Error: Required column '{req}' not found in IRA data")
            return pd.DataFrame()
            
    # Create DISTRITO_NORM
    df['DISTRITO_NORM'] = df['distrito'].apply(clean_distrito)
    print("DISTRITO_NORM created. Head:", df['DISTRITO_NORM'].head())
    
    # Create MES
    def week_to_month(w):
        try:
            w = int(w)
            m = (w - 1) // 4 + 1
            return min(m, 12)
        except:
            return 1
    df['MES'] = df['semana'].apply(week_to_month)
    
    # Create ANIO
    if 'ano' in df.columns:
        df['ANIO'] = df['ano']
    elif 'año' in df.columns:
        df['ANIO'] = df['año']
    else:
        print("Error: Year column (ano/año) not found")
        return pd.DataFrame()
        
    df['ANIO'] = pd.to_numeric(df['ANIO'], errors='coerce').fillna(0).astype(int)
    
    # Columns to sum
    cols_to_sum = ['tot_conf', 'neumonias_menores_5', 'hospitalizados', 'defunciones']
    existing_cols = [c for c in cols_to_sum if c in df.columns]
    print(f"Summing columns: {existing_cols}")
    
    # Convert to numeric
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Groupby
    group_cols = ['DISTRITO_NORM', 'ANIO', 'MES']
    
    print("Grouping by:", group_cols)
    print("Columns in DF:", df.columns.tolist())
    
    try:
        df_grouped = df.groupby(group_cols)[existing_cols].sum().reset_index()
        print("Groupby success!")
    except Exception as e:
        print(f"Error during groupby: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    print(f"IRA processed. Shape: {df_grouped.shape}")
    return df_grouped

if __name__ == "__main__":
    process_ira("IRA.xlsx")
