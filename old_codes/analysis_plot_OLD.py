import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

MASTER_FILE = "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"

def load_data():
    if not os.path.exists(MASTER_FILE):
        print(f"Error: {MASTER_FILE} not found.")
        return None
    return pd.read_csv(MASTER_FILE)

def plot_scatter(df):
    print("Generating Scatter Plot...")
    plt.figure(figsize=(10, 6))
    
    # Filter out 0 production for log scale (or add small constant)
    # User asked for Log Scale on X (Production)
    df_plot = df[df['PRODUCCION'] > 0].copy()
    
    if df_plot.empty:
        print("Warning: No data with Production > 0 for scatter plot.")
        return

    sns.scatterplot(
        data=df_plot,
        x='PRODUCCION',
        y='TASA_INCIDENCIA',
        hue='ESTRATO',
        alpha=0.6
    )
    
    plt.xscale('log')
    plt.title('Correlación: Producción Minera vs Tasa Incidencia IRA')
    plt.xlabel('Producción Minera (Escala Log)')
    plt.ylabel('Tasa de Incidencia (por 10k hab)')
    plt.tight_layout()
    plt.savefig('04_scatter_correlation.png')
    print("Saved 04_scatter_correlation.png")
    plt.close()

def plot_dual_axis(df):
    print("Generating Dual Axis Chart...")
    
    # Top 3 Mining Districts by Total Production
    top_districts = df.groupby('DISTRITO_NORM')['PRODUCCION'].sum().nlargest(3).index.tolist()
    
    if not top_districts:
        print("Warning: No districts found for dual axis plot.")
        return

    for district in top_districts:
        dist_data = df[df['DISTRITO_NORM'] == district].copy()
        
        # Aggregate by Month-Year for the plot
        # We need a date column
        dist_data['FECHA'] = pd.to_datetime(dist_data['ANIO'].astype(str) + '-' + dist_data['MES'].astype(str) + '-01')
        dist_data = dist_data.sort_values('FECHA')
        
        # Group by Date to handle multiple strata per month
        dist_grouped = dist_data.groupby('FECHA').agg({
            'PRODUCCION': 'sum',
            'TASA_INCIDENCIA': 'mean' # Incidence is constant per district-month
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Line chart for Incidence (Primary Axis - Focus on Health)
        color = 'tab:red'
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Tasa Incidencia IRA (por 1000 hab)', color=color)
        ax1.plot(dist_grouped['FECHA'], dist_grouped['TASA_INCIDENCIA'], color=color, marker='o', label='Incidencia', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart for Production (Secondary Axis)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Producción Minera', color=color)
        ax2.bar(dist_grouped['FECHA'], dist_grouped['PRODUCCION'], color=color, alpha=0.3, width=20, label='Producción')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)
        
        plt.title(f'Dinámica Temporal: Incidencia IRA vs Producción Minera - {district}')
        plt.tight_layout()
        filename = f'05_dual_axis_{district}.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_boxplot(df):
    print("Generating Boxplot...")
    
    # Create groups: Mining vs Non-Mining
    # We define "Mining District" as one that has ANY production ever? 
    # Or per month?
    # User said: "Divide los datos en dos grupos: 'Distritos Mineros' (Producción > 0) vs 'Distritos No Mineros' (Producción = 0)."
    # This implies row-level classification.
    
    df['TIPO_DISTRITO'] = np.where(df['PRODUCCION'] > 0, 'Minero', 'No Minero')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

MASTER_FILE = "03_MASTER_DATASET_ANALISIS_IMPACTO.csv"

def plot_scatter(df):
    print("Generating Scatter Plot...")
    plt.figure(figsize=(10, 6))
    
    # Filter out 0 production for log scale (or add small constant)
    # User asked for Log Scale on X (Production)
    df_plot = df[df['PRODUCCION'] > 0].copy()
    
    if df_plot.empty:
        print("Warning: No data with Production > 0 for scatter plot.")
        return

    sns.scatterplot(
        data=df_plot,
        x='PRODUCCION',
        y='TASA_INCIDENCIA',
        hue='ESTRATO',
        alpha=0.6
    )
    
    plt.xscale('log')
    plt.title('Correlación: Producción Minera vs Tasa Incidencia IRA')
    plt.xlabel('Producción Minera (Escala Log)')
    plt.ylabel('Tasa de Incidencia (por 10k hab)')
    plt.tight_layout()
    plt.savefig('04_scatter_correlation.png')
    print("Saved 04_scatter_correlation.png")
    plt.close()

def plot_dual_axis(df):
    print("Generating Dual Axis Chart...")
    
    # Top 3 Mining Districts by Total Production
    top_districts = df.groupby('DISTRITO_NORM')['PRODUCCION'].sum().nlargest(3).index.tolist()
    
    if not top_districts:
        print("Warning: No districts found for dual axis plot.")
        return

    for district in top_districts:
        dist_data = df[df['DISTRITO_NORM'] == district].copy()
        
        # Aggregate by Month-Year for the plot
        # We need a date column
        dist_data['FECHA'] = pd.to_datetime(dist_data['ANIO'].astype(str) + '-' + dist_data['MES'].astype(str) + '-01')
        dist_data = dist_data.sort_values('FECHA')
        
        # Group by Date to handle multiple strata per month
        dist_grouped = dist_data.groupby('FECHA').agg({
            'PRODUCCION': 'sum',
            'TASA_INCIDENCIA': 'mean' # Incidence is constant per district-month
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Line chart for Incidence (Primary Axis - Focus on Health)
        color = 'tab:red'
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Tasa Incidencia IRA (por 1000 hab)', color=color)
        ax1.plot(dist_grouped['FECHA'], dist_grouped['TASA_INCIDENCIA'], color=color, marker='o', label='Incidencia', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart for Production (Secondary Axis)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Producción Minera', color=color)
        ax2.bar(dist_grouped['FECHA'], dist_grouped['PRODUCCION'], color=color, alpha=0.3, width=20, label='Producción')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)
        
        plt.title(f'Dinámica Temporal: Incidencia IRA vs Producción Minera - {district}')
        plt.tight_layout()
        filename = f'05_dual_axis_{district}.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

def plot_boxplot(df):
    print("Generating Boxplot...")
    
    # Create groups: Mining vs Non-Mining
    # We define "Mining District" as one that has ANY production ever? 
    # Or per month?
    # User said: "Divide los datos en dos grupos: 'Distritos Mineros' (Producción > 0) vs 'Distritos No Mineros' (Producción = 0)."
    # This implies row-level classification.
    
    df['TIPO_DISTRITO'] = np.where(df['PRODUCCION'] > 0, 'Minero', 'No Minero')
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='TIPO_DISTRITO', y='TASA_INCIDENCIA', data=df)
    plt.title('Distribución de Tasa de Incidencia: Mineros vs No Mineros')
    plt.ylabel('Tasa de Incidencia')
    plt.tight_layout()
    plt.savefig('06_boxplot_comparativo.png')
    print("Saved 06_boxplot_comparativo.png")
    plt.close()

def plot_comparative_trend(df):
    print("Generating Comparative Trend Chart...")
    
    # Determine Geo Key
    if 'DISTRITO_NORM' in df.columns:
        geo_key = 'DISTRITO_NORM'
    elif 'PROVINCIA_NORM' in df.columns:
        geo_key = 'PROVINCIA_NORM'
    else:
        print("Error: No geographic key found for comparative plot.")
        return

    # Identify Mining vs Non-Mining Entities
    # Mining Entity: Has PRODUCCION > 0 at least once
    mining_entities = df[df['PRODUCCION'] > 0][geo_key].unique()
    
    df['TYPE'] = df[geo_key].apply(lambda x: 'Minero' if x in mining_entities else 'No Minero')
    
    # Create Date Column
    df['FECHA'] = pd.to_datetime(df['ANIO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    
    # Aggregate Incidence by Type and Date
    # We want the average incidence rate of districts in that group for that month
    trend = df.groupby(['FECHA', 'TYPE'])['TASA_INCIDENCIA'].mean().reset_index()
    
    # Pivot for plotting
    trend_pivot = trend.pivot(index='FECHA', columns='TYPE', values='TASA_INCIDENCIA')
    
    plt.figure(figsize=(12, 6))
    
    if 'Minero' in trend_pivot.columns:
        plt.plot(trend_pivot.index, trend_pivot['Minero'], label='Distritos Mineros (Expuestos)', color='tab:red', linewidth=2.5)
    if 'No Minero' in trend_pivot.columns:
        plt.plot(trend_pivot.index, trend_pivot['No Minero'], label='Distritos No Mineros (Control)', color='tab:blue', linestyle='--', linewidth=2)
        
    plt.title('Comparativo de Tendencias: Tasa de Incidencia IRA (Promedio Mensual)')
    plt.xlabel('Fecha')
    plt.ylabel('Tasa de Incidencia Promedio (por 1000 hab)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = '07_comparative_trend.png'
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def main():
    print("Loading Master Dataset...")
    try:
        df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")
    except Exception as e:
        print(f"Error loading master dataset: {e}")
        return

    # 1. Scatter Plot
    plot_scatter(df)
    
    # 2. Dual Axis (Only if District level, or adapt for Province)
    if 'DISTRITO_NORM' in df.columns:
        plot_dual_axis(df)
    else:
        print("Skipping Dual Axis (District level only).")
        
    # 3. Boxplot
    plot_boxplot(df)
    
    # 4. Comparative Trend
    plot_comparative_trend(df)
    
    print("Visualization Complete.")

if __name__ == "__main__":
    main()
