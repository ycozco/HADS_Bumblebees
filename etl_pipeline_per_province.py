import pandas as pd
import numpy as np
import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURACION
FILES = {
    "POBLACION": "POBLACION.xlsx",
    "IRA": "IRA.xlsx",
    "METALES": "COMPILADO METALES.xlsx",
    "NO_METALES": "COMPILADO NO METALES.xlsx"
}
OUTPUT_IMG_DIR = "images_province"
sns.set(style="whitegrid")

# --- FUNCIONES DE LIMPIEZA ---
def clean_name(name):
    """Normaliza nombres: mayusculas, sin tildes, N por N."""
    if pd.isna(name): return "UNKNOWN"
    return unidecode.unidecode(str(name).upper().strip()).replace('Ñ', 'N')

# --- ETL POR PROVINCIA ---
def process_population_prov(path):
    print(f"Procesando Poblacion (Provincia): {path}")
    try:
        df = pd.read_excel(path)
        year_cols = [c for c in df.columns if str(c).isdigit() and int(c) >= 2018]
        df = df.melt(id_vars=['Province'], value_vars=year_cols, var_name='ANIO', value_name='POBLACION')
        df['ANIO'] = df['ANIO'].astype(int)
        df['PROVINCIA_NORM'] = df['Province'].apply(clean_name)
        # Agrupar por Provincia y Año
        return df.groupby(['PROVINCIA_NORM', 'ANIO'])['POBLACION'].sum().reset_index()
    except Exception as e:
        print(f"Error Poblacion: {e}")
        return pd.DataFrame()

def process_ira_prov(path):
    print(f"Procesando IRA (Provincia): {path}")
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        if 'provincia' not in df.columns:
            print("Error: Columna 'provincia' no encontrada en IRA")
            return pd.DataFrame()
            
        df['PROVINCIA_NORM'] = df['provincia'].apply(clean_name)
        df['MES'] = df['semana'].apply(lambda w: min((int(w)-1)//4 + 1, 12) if pd.notna(w) else 1)
        df['ANIO'] = pd.to_numeric(df['ano'] if 'ano' in df.columns else df['año'], errors='coerce').fillna(0).astype(int)
        
        cols = ['ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas']
        valid_cols = [c for c in cols if c in df.columns]
        for c in valid_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Agrupar por Provincia, Año, Mes
        grouped = df.groupby(['PROVINCIA_NORM', 'ANIO', 'MES'])[valid_cols].sum().reset_index()
        grouped['TOTAL_CASOS'] = grouped[valid_cols].sum(axis=1)
        return grouped
    except Exception as e:
        print(f"Error IRA: {e}")
        return pd.DataFrame()

def process_mining_prov(path_met, path_non):
    print("Procesando Mineria (Provincia)...")
    def _proc(path):
        try:
            df = pd.read_excel(path, header=1)
            df.columns = [str(c).upper().strip() for c in df.columns]
            month_map = {'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SETIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12,'ENE':1,'FEB':2,'MAR':3,'ABR':4,'MAY':5,'JUN':6,'JUL':7,'AGO':8,'SET':9,'OCT':10,'NOV':11,'DIC':12,'SEPTIEMBRE':9}
            months = [m for m in month_map if m in df.columns]
            prov_col = next((c for c in df.columns if 'PROVINCIA' in c), None)
            
            if not prov_col or not months: return pd.DataFrame()
            
            id_vars = [c for c in ['ESTRATO', 'AÑO', prov_col] if c in df.columns]
            df = df.melt(id_vars=id_vars, value_vars=months, var_name='MES_STR', value_name='PRODUCCION')
            df['MES'] = df['MES_STR'].map(month_map)
            df['PROVINCIA_NORM'] = df[prov_col].apply(clean_name)
            df['PRODUCCION'] = pd.to_numeric(df['PRODUCCION'], errors='coerce').fillna(0)
            if 'AÑO' in df.columns: df.rename(columns={'AÑO':'ANIO'}, inplace=True)
            return df
        except Exception as e:
            print(f"Error Mineria {path}: {e}")
            return pd.DataFrame()

    df = pd.concat([_proc(path_met), _proc(path_non)], ignore_index=True)
    # Agrupar por Provincia, Año, Mes, Estrato
    return df.groupby(['PROVINCIA_NORM', 'ANIO', 'MES', 'ESTRATO'])['PRODUCCION'].sum().reset_index() if not df.empty else pd.DataFrame()

# --- VISUALIZACION ---
def save_plot(name):
    path = os.path.join(OUTPUT_IMG_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")

def plot_scatter(df):
    print("Generando Scatter (Provincia)...")
    data = df[df['PRODUCCION'] > 0]
    if data.empty: return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='PRODUCCION', y='TASA_INCIDENCIA', hue='ESTRATO', alpha=0.6)
    plt.xscale('log')
    plt.title('Correlacion (Provincia): Produccion vs Incidencia IRA')
    save_plot('01_scatter_provincia.png')

def plot_boxplot(df):
    print("Generando Boxplot (Provincia)...")
    # Clasificar Provincias como Mineras si tienen produccion alguna vez
    mining_provs = df[df['PRODUCCION'] > 0]['PROVINCIA_NORM'].unique()
    df['TIPO'] = df['PROVINCIA_NORM'].apply(lambda x: 'Minera' if x in mining_provs else 'No Minera')
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='TIPO', y='TASA_INCIDENCIA', data=df)
    plt.title('Incidencia: Provincias Mineras vs No Mineras')
    save_plot('02_boxplot_provincia.png')

def plot_trend(df):
    print("Generando Trend (Provincia)...")
    mining_provs = df[df['PRODUCCION'] > 0]['PROVINCIA_NORM'].unique()
    df['TIPO'] = df['PROVINCIA_NORM'].apply(lambda x: 'Minera' if x in mining_provs else 'No Minera')
    df['FECHA'] = pd.to_datetime(df['ANIO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    
    trend = df.groupby(['FECHA', 'TIPO'])['TASA_INCIDENCIA'].mean().unstack()
    plt.figure(figsize=(12, 6))
    if 'Minera' in trend: plt.plot(trend.index, trend['Minera'], 'r-', label='Mineras')
    if 'No Minera' in trend: plt.plot(trend.index, trend['No Minera'], 'b--', label='No Mineras')
    plt.title('Tendencia Promedio Incidencia IRA (Por Provincia)')
    plt.legend()
    save_plot('03_trend_provincia.png')

def plot_dual_axis_prov(df):
    print("Generando Dual Axis (Top 5 Provincias)...")
    # Top 5 Provincias por Produccion Total
    top_provs = df.groupby('PROVINCIA_NORM')['PRODUCCION'].sum().nlargest(5).index.tolist()
    
    for prov in top_provs:
        d = df[df['PROVINCIA_NORM'] == prov].copy()
        d['FECHA'] = pd.to_datetime(d['ANIO'].astype(str) + '-' + d['MES'].astype(str) + '-01')
        
        # Agrupar por fecha
        g = d.groupby('FECHA').agg({'PRODUCCION':'sum', 'TASA_INCIDENCIA':'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Incidencia IRA (x1000)', color='tab:red')
        ax1.plot(g['FECHA'], g['TASA_INCIDENCIA'], color='tab:red', marker='o', linewidth=2, label='Incidencia')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Produccion', color='tab:blue')
        ax2.bar(g['FECHA'], g['PRODUCCION'], color='tab:blue', alpha=0.3, width=20, label='Produccion')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.grid(False)
        
        plt.title(f'Dinamica Temporal: Provincia {prov}')
        save_plot(f'04_dual_axis_{prov}.png')

def main():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 1. ETL
    pop = process_population_prov(FILES["POBLACION"])
    ira = process_ira_prov(FILES["IRA"])
    mineria = process_mining_prov(FILES["METALES"], FILES["NO_METALES"])
    
    if pop.empty or ira.empty or mineria.empty: return print("Error: Datos vacios.")

    # 2. Merge Salud
    salud = pd.merge(ira, pop, on=['PROVINCIA_NORM', 'ANIO'], how='left')
    salud['TASA_INCIDENCIA'] = np.where((salud['POBLACION']>0) & (salud['TOTAL_CASOS'].notna()), (salud['TOTAL_CASOS']/salud['POBLACION'])*1000, np.nan)
    
    # 3. Master Merge
    keys = ['PROVINCIA_NORM', 'ANIO', 'MES']
    master = pd.merge(salud, mineria, on=keys, how='outer')
    master['PRODUCCION'] = master['PRODUCCION'].fillna(0)
    master['ESTRATO'] = master['ESTRATO'].fillna("NO MINERIA")
    
    print(f"Master Provincia Shape: {master.shape}")
    
    # 4. Plots
    plot_scatter(master)
    plot_boxplot(master)
    plot_trend(master)
    plot_dual_axis_prov(master)
    
    print("Proceso por Provincia Completado.")

if __name__ == "__main__":
    main()
