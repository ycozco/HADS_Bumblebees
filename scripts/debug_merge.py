import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "data", "csv")

HEALTH_FILE = os.path.join(DATA_CSV, "01_data_limpia_salud_mensual.csv")
MINING_FILE = os.path.join(DATA_CSV, "02_data_limpia_mineria_unificada.csv")

def debug_merge():
    print("Loading data...")
    health = pd.read_csv(HEALTH_FILE)
    mining = pd.read_csv(MINING_FILE)
    
    print("\n--- HEALTH DATA STATS ---")
    print(f"Total Cases (tot_conf): {health['tot_conf'].sum()}")
    print(f"Max Cases: {health['tot_conf'].max()}")
    print(f"Rows with Cases > 0: {len(health[health['tot_conf'] > 0])}")
    
    print("\n--- MINING DATA STATS ---")
    print(f"Total Production: {mining['PRODUCCION'].sum()}")
    print(f"Max Production: {mining['PRODUCCION'].max()}")
    print(f"Rows with Production > 0: {len(mining[mining['PRODUCCION'] > 0])}")
    
    print("\n--- MERGE TEST ---")
    # Try merging just on District first
    merged_dist = pd.merge(health, mining, on='DISTRITO_NORM', how='inner')
    print(f"Merge on District only: {len(merged_dist)} rows")
    
    # Try merging on District + Year
    merged_year = pd.merge(health, mining, on=['DISTRITO_NORM', 'ANIO'], how='inner')
    print(f"Merge on District + Year: {len(merged_year)} rows")
    
    # Try merging on District + Year + Month
    merged_full = pd.merge(health, mining, on=['DISTRITO_NORM', 'ANIO', 'MES'], how='inner')
    print(f"Merge on District + Year + Month: {len(merged_full)} rows")
    
    if not merged_full.empty:
        print("\nSample full merge:")
        print(merged_full[['DISTRITO_NORM', 'ANIO', 'MES', 'tot_conf', 'PRODUCCION']].head())
        
        both_pos = merged_full[(merged_full['tot_conf'] > 0) & (merged_full['PRODUCCION'] > 0)]
        print(f"\nRows with both > 0: {len(both_pos)}")
        if not both_pos.empty:
            print(both_pos.head())
    else:
        print("\nFull merge is empty!")
        # Check why
        # Pick a district present in both
        common_dist = set(health['DISTRITO_NORM']).intersection(set(mining['DISTRITO_NORM']))
        if common_dist:
            d = list(common_dist)[0]
            print(f"\nInspecting District: {d}")
            print("Health entries:")
            print(health[health['DISTRITO_NORM'] == d][['ANIO', 'MES']].head())
            print("Mining entries:")
            print(mining[mining['DISTRITO_NORM'] == d][['ANIO', 'MES']].head())

if __name__ == "__main__":
    debug_merge()
