import pandas as pd
import numpy as np
import unidecode
import os

# --- CONFIGURATION ---
FILES = {
    "POBLACION": "POBLACION.xlsx",
    "IRA": "IRA.xlsx",
    "METALES": "COMPILADO METALES.xlsx",
    "NO_METALES": "COMPILADO NO METALES.xlsx"
}

OUTPUTS = {
    "SALUD_CLEAN": "01_data_limpia_salud_mensual.csv",
    "MINERIA_CLEAN": "02_data_limpia_mineria_unificada.csv",
    "MASTER": "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"
}

# --- HELPER FUNCTIONS ---

# --- HELPER FUNCTIONS ---

def clean_distrito(name):
    if pd.isna(name):
        return "UNKNOWN"
    # 1. String conversion
    name = str(name)
    # 2. Uppercase
    name = name.upper()
    # 3. Strip
    name = name.strip()
    # 4. Unidecode (removes accents)
    name = unidecode.unidecode(name)
    # 5. Replace Ñ with N (just in case unidecode missed it or for safety)
    name = name.replace('Ñ', 'N')
    return name

def process_population(file_path):
    print(f"Processing Population data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Population file: {e}")
        return pd.DataFrame()
    
    year_cols = [c for c in df.columns if str(c).isdigit() and int(c) >= 2018]
    
    df_melted = pd.melt(
        df, 
        id_vars=['District'], 
        value_vars=year_cols,
        var_name='ANIO', 
        value_name='POBLACION'
    )
    
    df_melted['DISTRITO_NORM'] = df_melted['District'].apply(clean_distrito)
    df_melted['ANIO'] = df_melted['ANIO'].astype(int)
    
    df_final = df_melted.groupby(['DISTRITO_NORM', 'ANIO'])['POBLACION'].sum().reset_index()
    print(f"Population processed. Shape: {df_final.shape}")
    return df_final

def process_ira(file_path):
    print(f"Processing IRA data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading IRA file: {e}")
        return pd.DataFrame()
    
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if 'distrito' not in df.columns:
        print("Error: 'distrito' column not found in IRA")
        return pd.DataFrame()
        
    df['DISTRITO_NORM'] = df['distrito'].apply(clean_distrito)
    
    def week_to_month(w):
        try:
            w = int(w)
            m = (w - 1) // 4 + 1
            return min(m, 12)
        except:
            return 1
            
    if 'semana' in df.columns:
        df['MES'] = df['semana'].apply(week_to_month)
    else:
        print("Error: 'semana' column not found")
        return pd.DataFrame()
    
    if 'ano' in df.columns:
        df['ANIO'] = df['ano']
    elif 'año' in df.columns:
        df['ANIO'] = df['año']
import pandas as pd
import numpy as np
import unidecode
import os

# --- CONFIGURATION ---
FILES = {
    "POBLACION": "POBLACION.xlsx",
    "IRA": "IRA.xlsx",
    "METALES": "COMPILADO METALES.xlsx",
    "NO_METALES": "COMPILADO NO METALES.xlsx"
}

OUTPUTS = {
    "SALUD_CLEAN": "01_data_limpia_salud_mensual.csv",
    "MINERIA_CLEAN": "02_data_limpia_mineria_unificada.csv",
    "MASTER": "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"
}

# --- HELPER FUNCTIONS ---

# --- HELPER FUNCTIONS ---

def clean_distrito(name):
    if pd.isna(name):
        return "UNKNOWN"
    # 1. String conversion
    name = str(name)
    # 2. Uppercase
    name = name.upper()
    # 3. Strip
    name = name.strip()
    # 4. Unidecode (removes accents)
    name = unidecode.unidecode(name)
    # 5. Replace Ñ with N (just in case unidecode missed it or for safety)
    name = name.replace('Ñ', 'N')
    return name

def process_population(file_path):
    print(f"Processing Population data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Population file: {e}")
import pandas as pd
import numpy as np
import unidecode
import os

# --- CONFIGURATION ---
FILES = {
    "POBLACION": "POBLACION.xlsx",
    "IRA": "IRA.xlsx",
    "METALES": "COMPILADO METALES.xlsx",
    "NO_METALES": "COMPILADO NO METALES.xlsx"
}

OUTPUTS = {
    "SALUD_CLEAN": "01_data_limpia_salud_mensual.csv",
    "MINERIA_CLEAN": "02_data_limpia_mineria_unificada.csv",
    "MASTER": "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"
}

# --- HELPER FUNCTIONS ---

# --- HELPER FUNCTIONS ---

def clean_distrito(name):
    if pd.isna(name):
        return "UNKNOWN"
    # 1. String conversion
    name = str(name)
    # 2. Uppercase
    name = name.upper()
    # 3. Strip
    name = name.strip()
    # 4. Unidecode (removes accents)
    name = unidecode.unidecode(name)
    # 5. Replace Ñ with N (just in case unidecode missed it or for safety)
    name = name.replace('Ñ', 'N')
    return name

def process_population(file_path):
    print(f"Processing Population data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Population file: {e}")
        return pd.DataFrame()
    
    year_cols = [c for c in df.columns if str(c).isdigit() and int(c) >= 2018]
    
    df_melted = pd.melt(
        df, 
        id_vars=['District', 'Province'], # Include Province
        value_vars=year_cols,
        var_name='ANIO', 
        value_name='POBLACION'
    )
    
    df_melted['DISTRITO_NORM'] = df_melted['District'].apply(clean_distrito)
    if 'Province' in df_melted.columns:
        df_melted['PROVINCIA_NORM'] = df_melted['Province'].apply(clean_distrito)
    else:
        df_melted['PROVINCIA_NORM'] = "UNKNOWN"
        
    df_melted['ANIO'] = df_melted['ANIO'].astype(int)
    
    # Group by District and Province
    df_final = df_melted.groupby(['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO'])['POBLACION'].sum().reset_index()
    print(f"Population processed. Shape: {df_final.shape}")
    return df_final

def process_ira(file_path):
    print(f"Processing IRA data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading IRA file: {e}")
        return pd.DataFrame()
    
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if 'distrito' not in df.columns:
        print("Error: 'distrito' column not found in IRA")
        return pd.DataFrame()
        
    df['DISTRITO_NORM'] = df['distrito'].apply(clean_distrito)
    
    if 'provincia' in df.columns:
        df['PROVINCIA_NORM'] = df['provincia'].apply(clean_distrito)
    else:
        df['PROVINCIA_NORM'] = "UNKNOWN"
    
    def week_to_month(w):
        try:
            w = int(w)
            m = (w - 1) // 4 + 1
            return min(m, 12)
        except:
            return 1
            
    if 'semana' in df.columns:
        df['MES'] = df['semana'].apply(week_to_month)
    else:
        print("Error: 'semana' column not found")
        return pd.DataFrame()
    
    if 'ano' in df.columns:
        df['ANIO'] = df['ano']
    elif 'año' in df.columns:
        df['ANIO'] = df['año']
    else:
        print("Error: Year column not found")
        return pd.DataFrame()
        
    df['ANIO'] = pd.to_numeric(df['ANIO'], errors='coerce').fillna(0).astype(int)
    
    # Columns to sum
    # Based on inspection:
    cols_to_sum = [
        'ira_no_neumonia', 
        'neumonias_men5', 
        'neumonias_60mas', 
        'hospitalizados_men5', 
        'hospitalizados_60mas', 
        'defunciones_men5', 
        'defunciones_60mas'
    ]
    
    existing_cols = [c for c in cols_to_sum if c in df.columns]
    
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    group_cols = ['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES']
    
    try:
        df_grouped = df.groupby(group_cols)[existing_cols].sum().reset_index()
    except Exception as e:
        print(f"Error during groupby: {e}")
        return pd.DataFrame()
    
    for col in cols_to_sum:
        if col not in df_grouped.columns:
            df_grouped[col] = 0
            
    # Calculate TOTAL_CASOS
    # We assume Total = IRA (no neumonia) + Neumonias
    df_grouped['TOTAL_CASOS'] = (
        df_grouped['ira_no_neumonia'] + 
        df_grouped['neumonias_men5'] + 
        df_grouped['neumonias_60mas']
    )
            
    print(f"IRA processed. Shape: {df_grouped.shape}")
    return df_grouped

def process_mining(metals_path, non_metals_path):
    print("Processing Mining data...")
    
    def process_one_mining_file(path, type_label):
        try:
            df = pd.read_excel(path, header=1)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return pd.DataFrame()
            
        df.columns = [str(c).upper().strip() for c in df.columns]
        
        # Support both full names and abbreviations
        month_map_full = {
            'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
            'JULIO': 7, 'AGOSTO': 8, 'SETIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12,
            'ENE': 1, 'FEB': 2, 'MAR': 3, 'ABR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AGO': 8, 'SET': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12,
            'SEPTIEMBRE': 9 # Handle variant
        }
        
        available_months = [m for m in month_map_full.keys() if m in df.columns]
        
        dist_col = None
        for c in df.columns:
            if 'DISTRITO' in c:
                dist_col = c
                break
        
        prov_col = None
        for c in df.columns:
            if 'PROVINCIA' in c:
                prov_col = c
                break
        
        if not dist_col:
            return pd.DataFrame()

        id_vars = ['ESTRATO', 'AÑO', dist_col]
        if prov_col:
            id_vars.append(prov_col)
            
        id_vars = [c for c in id_vars if c in df.columns]
        
        if not available_months:
            return pd.DataFrame()
            
        df_melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=available_months,
            var_name='MES_STR',
            value_name='PRODUCCION'
        )
        
        df_melted['MES'] = df_melted['MES_STR'].map(month_map_full)
        
        df_melted['DISTRITO_NORM'] = df_melted[dist_col].apply(clean_distrito)
        if prov_col:
            df_melted['PROVINCIA_NORM'] = df_melted[prov_col].apply(clean_distrito)
        else:
            df_melted['PROVINCIA_NORM'] = "UNKNOWN"
            
        df_melted['PRODUCCION'] = pd.to_numeric(df_melted['PRODUCCION'], errors='coerce').fillna(0)
        
        if 'AÑO' in df_melted.columns:
            df_melted.rename(columns={'AÑO': 'ANIO'}, inplace=True)
            
        return df_melted

    df_metals = process_one_mining_file(metals_path, "METALES")
    df_non_metals = process_one_mining_file(non_metals_path, "NO_METALES")
    
    df_combined = pd.concat([df_metals, df_non_metals], ignore_index=True)
    
    if df_combined.empty:
        print("Warning: Mining data is empty")
        return pd.DataFrame()
        
    df_grouped = df_combined.groupby(['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES', 'ESTRATO'])['PRODUCCION'].sum().reset_index()
    
    print(f"Mining processed. Shape: {df_grouped.shape}")
    return df_grouped

def main():
    # 1. Process Population
    pop_df = process_population(FILES["POBLACION"])
    
    # 2. Process IRA
    ira_df = process_ira(FILES["IRA"])
    
    if ira_df.empty or pop_df.empty:
        print("Critical Error: IRA or Population data empty. Aborting.")
        return

    # 3. Merge IRA + Population
    print("Merging Health and Population...")
    # Merge on District, Province, Year
    health_pop = pd.merge(ira_df, pop_df, on=['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO'], how='left')
    
    print("Calculating Incidence Rate (per 1000)...")
    # Strict NaN handling: If Population is NaN, Incidence is NaN.
    # If TOTAL_CASOS is NaN, Incidence is NaN.
    # We only fillna(0) for Population if we are sure? No, better to leave as is if missing.
    # But Population data should be complete for all districts in the list?
    # Let's assume if Population is missing, we can't calculate rate.
    
    health_pop['TASA_INCIDENCIA'] = np.where(
        (health_pop['POBLACION'] > 0) & (health_pop['TOTAL_CASOS'].notna()),
        (health_pop['TOTAL_CASOS'] / health_pop['POBLACION']) * 1000, 
        np.nan
    )
    
    health_pop.to_csv(OUTPUTS["SALUD_CLEAN"], index=False, encoding='utf-8')
    print(f"Saved {OUTPUTS['SALUD_CLEAN']}")
    
    # 4. Process Mining
    mining_df = process_mining(FILES["METALES"], FILES["NO_METALES"])
    
    if mining_df.empty:
        print("Critical Error: Mining data empty. Aborting.")
        return

    mining_df.to_csv(OUTPUTS["MINERIA_CLEAN"], index=False, encoding='utf-8')
    print(f"Saved {OUTPUTS['MINERIA_CLEAN']}")
    
    # --- DIAGNOSTICS: Orphan Districts ---
    print("\n--- DIAGNOSTICS: Orphan Districts ---")
    health_districts = set(health_pop['DISTRITO_NORM'].unique())
    mining_districts = set(mining_df['DISTRITO_NORM'].unique())
    
    orphans = mining_districts - health_districts
    print(f"Mining Districts NOT in Health Data (Orphans): {len(orphans)}")
    if orphans:
        print("List of Orphan Districts (Top 20):")
        print(sorted(list(orphans))[:20])
        
    intersection = health_districts.intersection(mining_districts)
    overlap_pct = len(intersection) / len(mining_districts) * 100 if mining_districts else 0
    print(f"Overlap Percentage: {overlap_pct:.2f}%")
    
    # --- CONTINGENCY: Province Aggregation ---
    use_province = False
    if overlap_pct < 80:
        print("\nWARNING: Low overlap (< 80%). Switching to PROVINCE aggregation as contingency.")
        use_province = True
        
        # Aggregate Health by Province
        health_prov = health_pop.groupby(['PROVINCIA_NORM', 'ANIO', 'MES']).agg({
            'TOTAL_CASOS': 'sum',
            'POBLACION': 'sum' # Summing population of districts? Yes.
        }).reset_index()
        
        health_prov['TASA_INCIDENCIA'] = np.where(
            health_prov['POBLACION'] > 0,
            (health_prov['TOTAL_CASOS'] / health_prov['POBLACION']) * 1000,
            np.nan
        )
        
        # Aggregate Mining by Province
        mining_prov = mining_df.groupby(['PROVINCIA_NORM', 'ANIO', 'MES', 'ESTRATO'])['PRODUCCION'].sum().reset_index()
        
        # Use these for Master
        left_df = health_prov
        right_df = mining_prov
        merge_keys = ['PROVINCIA_NORM', 'ANIO', 'MES']
        print("Using PROVINCE level data for Master Dataset.")
    else:
        left_df = health_pop
        right_df = mining_df
        merge_keys = ['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES']
        print("Using DISTRICT level data for Master Dataset.")
        
    # 5. Create Master Dataset (Outer Join)
    print("\nCreating Master Dataset (Outer Join)...")
    master = pd.merge(left_df, right_df, on=merge_keys, how='outer', indicator=True)
    
    print("Merge Status Counts:")
    print(master['_merge'].value_counts())
    
    # Fill NaN - STRICT
    # Only fill Production with 0
    master['PRODUCCION'] = master['PRODUCCION'].fillna(0)
    master['ESTRATO'] = master['ESTRATO'].fillna("NO MINERIA")
    
    # Do NOT fill Health data with 0.
    
    # Export Master
    master.to_csv(OUTPUTS["MASTER"], index=False, encoding='utf-8')
    print(f"Saved {OUTPUTS['MASTER']}")
    
    # --- VERIFICATION ---
    print("\n--- VERIFICATION: Sample Rows (TOTAL_CASOS > 0 AND PRODUCCION > 0) ---")
    sample = master[(master['TOTAL_CASOS'] > 0) & (master['PRODUCCION'] > 0)]
    
    if sample.empty:
        print("CRITICAL ERROR: No rows found with both IRA cases and Mining Production > 0.")
    else:
        cols_to_show = merge_keys + ['TOTAL_CASOS', 'PRODUCCION', 'TASA_INCIDENCIA']
        print(sample[cols_to_show].head(5))

    print("ETL Complete.")

if __name__ == "__main__":
    main()
