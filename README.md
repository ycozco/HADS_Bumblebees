# INFORME TÉCNICO: Análisis de Impacto Minero en Salud Respiratoria (Arequipa)

## 1. Objetivo del ETL
El objetivo principal de este proceso ETL (Extract, Transform, Load) fue unificar tres fuentes de datos dispares para analizar la correlación entre la actividad minera y la incidencia de enfermedades respiratorias (IRA).
- **Desafío Principal**: La granularidad temporal y espacial no coincidía.
    - **Salud (IRA)**: Reportado semanalmente (Semana Epidemiológica) por establecimiento de salud.
    - **Minería**: Reportado mensualmente por titular minero.
    - **Población**: Datos anuales por distrito.
- **Solución**: Se construyó un pipeline que agrega los datos de salud a nivel mensual y normaliza los nombres de los distritos para permitir el cruce con los datos de producción minera.

## 2. Desafío de Calidad de Datos
Durante la integración, nos enfrentamos a severos problemas de calidad de datos en las llaves de cruce (nombres de distritos).

### Problemas Identificados
1.  **Inconsistencia de Nombres**:
    -   `YURA` vs `DISTRITO DE YURA` vs `YURA ` (espacios).
    -   Caracteres especiales: `CAYLLOMA` vs `CAYLLOMA`.
2.  **Distritos Huérfanos**: Inicialmente, un porcentaje significativo de distritos mineros no cruzaba con los datos de salud debido a estas discrepancias.
3.  **Datos Faltantes (NaN)**: El manejo incorrecto de ceros podía sesgar el análisis (asumir 0 casos cuando no hay datos).

### Soluciones Implementadas
-   **Limpieza Agresiva de Strings**: Se implementó una función de normalización estricta:
    ```python
    def clean_distrito(name):
        name = unidecode.unidecode(str(name).upper().strip())
        return name.replace('Ñ', 'N')
    ```
-   **Diagnóstico de Huérfanos**: Se añadió un paso de auditoría automática que alerta si el solapamiento de distritos es menor al 80%.
-   **Manejo Estricto de NaN**:
    -   **Producción**: Se rellenó con 0 (asumiendo que sin reporte = sin producción).
    -   **Salud**: Se mantuvieron los `NaN` si no había reporte, para no diluir artificialmente la tasa de incidencia.

## 3. Metodología de Normalización
Para comparar distritos con poblaciones muy diferentes (ej: Cerro Colorado vs. San Juan de Tarucani), no se puede usar el número absoluto de casos.

**Fórmula de Tasa de Incidencia:**
$$ \text{Tasa} = \left( \frac{\text{Total Casos IRA}}{\text{Población}} \right) \times 1000 $$

-   **Total Casos IRA**: Suma de `ira_no_neumonia` + `neumonias_men5` + `neumonias_60mas`.
-   **Interpretación**: Número de casos por cada 1,000 habitantes.

## 4. Interpretación de Resultados

### Correlación Visual (Scatter Plot)
El gráfico de dispersión (`04_scatter_correlation.png`) muestra la relación entre el volumen de producción (escala logarítmica) y la tasa de incidencia.
-   **Observación**: Se busca identificar si a mayor producción, mayor incidencia. La dispersión sugiere que la relación no es lineal simple y puede depender de otros factores (altura, temperatura).

### Dinámica Temporal (Dual Axis)
Los gráficos de doble eje (`05_dual_axis_[DISTRITO].png`) para los distritos top (Yura, Uchumayo, etc.) permiten ver la estacionalidad.
-   **Patrón**: Las curvas de IRA suelen mostrar picos estacionales (invierno), independientemente de la producción minera constante. Esto sugiere que el clima es un factor dominante.

### Comparativo de Tendencias (Control vs. Expuesto)
El gráfico `07_comparative_trend.png` es la prueba ácida. Compara la tasa promedio de todos los distritos mineros vs. los no mineros.
-   **Hipótesis del Offset**: Si la línea roja (Mineros) está sistemáticamente por encima de la azul (No Mineros), sugiere un riesgo base más alto en zonas mineras, más allá del efecto estacional.

## 5. Conclusiones y Recomendaciones

### Conclusiones
1.  **Integración Exitosa**: Se logró unificar datos heterogéneos con un alto porcentaje de coincidencia (>98%) a nivel distrital.
2.  **Estacionalidad Dominante**: La incidencia de IRA sigue un fuerte patrón estacional, lo que obliga a usar grupos de control para aislar el efecto minero.

### Recomendaciones
1.  **Incorporar Datos Ambientales**: Para probar causalidad, es indispensable cruzar estos datos con mediciones de Calidad de Aire (PM10, PM2.5) y Meteorología (Temperatura, Viento).
2.  **Análisis de Rezagos (Lags)**: Investigar si el aumento en producción tiene un efecto retardado (ej: 1-2 meses después) en la salud.
3.  **Granularidad**: Intentar obtener datos de salud semanales limpios para un análisis más fino de brotes.
