import pandas as pd
import numpy as np
import unidecode
import os

# CONFIGURACION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_EXCEL = os.path.join(BASE_DIR, "data", "excel")
DATA_CSV = os.path.join(BASE_DIR, "data", "csv")

FILES = {
    "POBLACION": os.path.join(DATA_EXCEL, "POBLACION.xlsx"),
    "IRA": os.path.join(DATA_EXCEL, "IRA.xlsx"),
    "METALES": os.path.join(DATA_EXCEL, "COMPILADO METALES.xlsx"),
    "NO_METALES": os.path.join(DATA_EXCEL, "COMPILADO NO METALES.xlsx")
}
OUTPUTS = {
    "SALUD": os.path.join(DATA_CSV, "01_data_limpia_salud_mensual.csv"),
    "MINERIA": os.path.join(DATA_CSV, "02_data_limpia_mineria_unificada.csv"),
    "MASTER": os.path.join(DATA_CSV, "03_MASTER_DATASET_ANALISIS_IMPACTO.csv")
}

# FUNCIONES AUXILIARES
def clean_distrito(name):
    """Normaliza nombres de distritos: mayusculas, sin tildes, N por N."""
    if pd.isna(name): return "UNKNOWN"
    return unidecode.unidecode(str(name).upper().strip()).replace('Ñ', 'N')

def process_population(path):
    """Procesa poblacion: derrite anios y normaliza distrito/provincia."""
    print(f"Procesando Poblacion: {path}")
    try:
        df = pd.read_excel(path)
        year_cols = [c for c in df.columns if str(c).isdigit() and int(c) >= 2018]
        df = df.melt(id_vars=['District', 'Province'], value_vars=year_cols, var_name='ANIO', value_name='POBLACION')
        df['ANIO'] = df['ANIO'].astype(int)
        df['DISTRITO_NORM'] = df['District'].apply(clean_distrito)
        df['PROVINCIA_NORM'] = df['Province'].apply(clean_distrito) if 'Province' in df.columns else "UNKNOWN"
        return df.groupby(['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO'])['POBLACION'].sum().reset_index()
    except Exception as e:
        print(f"Error Poblacion: {e}")
        return pd.DataFrame()

def process_ira(path):
    """Procesa IRA: suma casos, agrupa por mes y normaliza."""
    print(f"Procesando IRA: {path}")
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        df['DISTRITO_NORM'] = df['distrito'].apply(clean_distrito)
        df['PROVINCIA_NORM'] = df['provincia'].apply(clean_distrito) if 'provincia' in df.columns else "UNKNOWN"
        df['MES'] = df['semana'].apply(lambda w: min((int(w)-1)//4 + 1, 12) if pd.notna(w) else 1)
        df['ANIO'] = pd.to_numeric(df['ano'] if 'ano' in df.columns else df['año'], errors='coerce').fillna(0).astype(int)
        
        cols = ['ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas', 'hospitalizados_men5', 'hospitalizados_60mas', 'defunciones_men5', 'defunciones_60mas']
        valid_cols = [c for c in cols if c in df.columns]
        for c in valid_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        grouped = df.groupby(['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES'])[valid_cols].sum().reset_index()
        grouped['TOTAL_CASOS'] = grouped[[c for c in ['ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas'] if c in grouped]].sum(axis=1)
        return grouped
    except Exception as e:
        print(f"Error IRA: {e}")
        return pd.DataFrame()

def process_mining(path_met, path_non):
    """Procesa Mineria: unifica metales/no metales, derrite meses y agrupa."""
    print("Procesando Mineria...")
    def _proc(path):
        try:
            df = pd.read_excel(path, header=1)
            df.columns = [str(c).upper().strip() for c in df.columns]
            month_map = {'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SETIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12,'ENE':1,'FEB':2,'MAR':3,'ABR':4,'MAY':5,'JUN':6,'JUL':7,'AGO':8,'SET':9,'OCT':10,'NOV':11,'DIC':12,'SEPTIEMBRE':9}
            months = [m for m in month_map if m in df.columns]
            dist_col = next((c for c in df.columns if 'DISTRITO' in c), None)
            prov_col = next((c for c in df.columns if 'PROVINCIA' in c), None)
            if not dist_col or not months: return pd.DataFrame()
            
            id_vars = [c for c in ['ESTRATO', 'AÑO', dist_col, prov_col] if c in df.columns]
            df = df.melt(id_vars=id_vars, value_vars=months, var_name='MES_STR', value_name='PRODUCCION')
            df['MES'] = df['MES_STR'].map(month_map)
            df['DISTRITO_NORM'] = df[dist_col].apply(clean_distrito)
            df['PROVINCIA_NORM'] = df[prov_col].apply(clean_distrito) if prov_col else "UNKNOWN"
            df['PRODUCCION'] = pd.to_numeric(df['PRODUCCION'], errors='coerce').fillna(0)
            if 'AÑO' in df.columns: df.rename(columns={'AÑO':'ANIO'}, inplace=True)
            return df
        except Exception as e:
            print(f"Error Mineria {path}: {e}")
            return pd.DataFrame()

    df = pd.concat([_proc(path_met), _proc(path_non)], ignore_index=True)
    return df.groupby(['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES', 'ESTRATO'])['PRODUCCION'].sum().reset_index() if not df.empty else pd.DataFrame()

def main():
    # 1. Procesar Datos
    pop = process_population(FILES["POBLACION"])
    ira = process_ira(FILES["IRA"])
    mineria = process_mining(FILES["METALES"], FILES["NO_METALES"])
    
    if pop.empty or ira.empty or mineria.empty: return print("Error: Datos vacios.")

    # 2. Merge Salud (IRA + Pob)
    salud = pd.merge(ira, pop, on=['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO'], how='left')
    salud['TASA_INCIDENCIA'] = np.where((salud['POBLACION']>0) & (salud['TOTAL_CASOS'].notna()), (salud['TOTAL_CASOS']/salud['POBLACION'])*1000, np.nan)
    salud.to_csv(OUTPUTS["SALUD"], index=False)

    # 3. Diagnostico Huerfanos
    orphans = set(mineria['DISTRITO_NORM']) - set(salud['DISTRITO_NORM'])
    overlap = len(set(salud['DISTRITO_NORM']) & set(mineria['DISTRITO_NORM'])) / len(set(mineria['DISTRITO_NORM'])) * 100
    print(f"Overlap Distritos: {overlap:.2f}%. Huerfanos: {len(orphans)}")

    # 4. Contingencia Provincia
    keys = ['DISTRITO_NORM', 'PROVINCIA_NORM', 'ANIO', 'MES']
    if overlap < 80:
        print("WARN: Bajo overlap. Usando PROVINCIA.")
        keys = ['PROVINCIA_NORM', 'ANIO', 'MES']
        salud = salud.groupby(keys).agg({'TOTAL_CASOS':'sum', 'POBLACION':'sum'}).reset_index()
        salud['TASA_INCIDENCIA'] = np.where(salud['POBLACION']>0, (salud['TOTAL_CASOS']/salud['POBLACION'])*1000, np.nan)
        mineria = mineria.groupby(keys + ['ESTRATO'])['PRODUCCION'].sum().reset_index()

    # 5. Master Dataset
    mineria.to_csv(OUTPUTS["MINERIA"], index=False)
    master = pd.merge(salud, mineria, on=keys, how='outer')
    master['PRODUCCION'] = master['PRODUCCION'].fillna(0)
    master['ESTRATO'] = master['ESTRATO'].fillna("NO MINERIA")
    master.to_csv(OUTPUTS["MASTER"], index=False)
    print(f"ETL Completado. Master guardado en {OUTPUTS['MASTER']}")

if __name__ == "__main__":
    main()
