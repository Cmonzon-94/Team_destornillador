import pandas as pd
import numpy as np

def describe_df(df):
    """
    Genera un resumen descriptivo de un DataFrame con información sobre cada columna.

    Argumentos:
    df (pd.DataFrame): El DataFrame que se desea analizar.

    Retorna:
    pd.DataFrame: Un DataFrame resumen con las siguientes filas para cada columna:
        - type: Tipo de datos de la columna.
        - null_percentage: Porcentaje de valores nulos.
        - unique_values: Número de valores únicos en la columna.
        - cardinality_percentage: Porcentaje de cardinalidad (valores únicos / total de filas).
    """
    summary = {
        "type": df.dtypes,
        "null_percentage": df.isnull().mean() * 100,
        "unique_values": df.nunique(),
        "cardinality_percentage": (df.nunique() / len(df)) * 100
    }

    summary_df = pd.DataFrame(summary)  # Crear un DataFrame con las métricas calculadas
    return summary_df

# Ejemplo de uso
if __name__ == "__main__":
    data = {
        "A": [1, 2, 3, 4, np.nan],
        "B": ["a", "b", "a", "c", "c"],
        "C": [1.5, np.nan, 3.5, 4.5, 5.5],
        "D": [1, 1, 1, 1, 0]
    }
    example_df = pd.DataFrame(data)

    # Usar describe_df
    result_describe = describe_df(example_df)
    print("Resumen Descriptivo:\n", result_describe)
    