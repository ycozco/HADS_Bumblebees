import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURACION
MASTER_FILE = "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"
IMG_DIR = "result_images"
sns.set(style="whitegrid")

def save_plot(name):
    """Guarda grafico en carpeta result_images."""
    path = os.path.join(IMG_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")

def plot_scatter(df):
    """Grafico dispersion: Produccion vs Incidencia."""
    print("Generando Scatter...")
    data = df[df['PRODUCCION'] > 0]
    if data.empty: return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='PRODUCCION', y='TASA_INCIDENCIA', hue='ESTRATO', alpha=0.6)
    plt.xscale('log')
    plt.title('Correlacion: Produccion vs Incidencia IRA')
    save_plot('04_scatter_correlation.png')

def plot_dual_axis(df):
    """Grafico doble eje: Produccion (Barras) e Incidencia (Linea) por tiempo."""
    print("Generando Dual Axis...")
    key = 'DISTRITO_NORM' if 'DISTRITO_NORM' in df.columns else 'PROVINCIA_NORM'
    # Lista fija solicitada por el usuario
    top = ['YURA', 'UCHUMAYO', 'SAN JUAN DE TARUCANI', 'CALLALLI', 'COCACHACRA']
    
    for entity in top:
        d = df[df[key] == entity].copy()
        d['FECHA'] = pd.to_datetime(d['ANIO'].astype(str) + '-' + d['MES'].astype(str) + '-01')
        g = d.groupby('FECHA').agg({'PRODUCCION':'sum', 'TASA_INCIDENCIA':'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Incidencia IRA (x1000)', color='tab:red')
        ax1.plot(g['FECHA'], g['TASA_INCIDENCIA'], color='tab:red', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Produccion', color='tab:blue')
        ax2.bar(g['FECHA'], g['PRODUCCION'], color='tab:blue', alpha=0.3, width=20)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        plt.title(f'Dinamica Temporal: {entity}')
        save_plot(f'05_dual_axis_{entity}.png')

def plot_boxplot(df):
    """Boxplot comparativo: Mineros vs No Mineros."""
    print("Generando Boxplot...")
    df['TIPO'] = np.where(df['PRODUCCION'] > 0, 'Minero', 'No Minero')
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='TIPO', y='TASA_INCIDENCIA', data=df)
    plt.title('Incidencia: Mineros vs No Mineros')
    save_plot('06_boxplot_comparativo.png')

def plot_trend(df):
    """Tendencia comparativa promedio mensual."""
    print("Generando Tendencia...")
    key = 'DISTRITO_NORM' if 'DISTRITO_NORM' in df.columns else 'PROVINCIA_NORM'
    mineros = df[df['PRODUCCION'] > 0][key].unique()
    df['TIPO'] = df[key].apply(lambda x: 'Minero' if x in mineros else 'No Minero')
    df['FECHA'] = pd.to_datetime(df['ANIO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    
    trend = df.groupby(['FECHA', 'TIPO'])['TASA_INCIDENCIA'].mean().unstack()
    plt.figure(figsize=(12, 6))
    if 'Minero' in trend: plt.plot(trend.index, trend['Minero'], 'r-', label='Mineros')
    if 'No Minero' in trend: plt.plot(trend.index, trend['No Minero'], 'b--', label='No Mineros')
    plt.title('Tendencia Promedio Incidencia IRA')
    plt.legend()
    save_plot('07_comparative_trend.png')

def main():
    if not os.path.exists(MASTER_FILE): return print("Error: No existe Master Dataset.")
    df = pd.read_csv(MASTER_FILE)
    plot_scatter(df)
    plot_dual_axis(df)
    plot_boxplot(df)
    plot_trend(df)
    print("Visualizacion Completa.")

if __name__ == "__main__":
    main()
