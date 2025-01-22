import pandas as pd

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las columnas de un DataFrame según su tipo sugerido basado en cardinalidad y umbrales.

    Argumentos:
    df (pd.DataFrame): El DataFrame que se desea analizar.
    umbral_categoria (int): Umbral máximo para considerar una variable como categórica.
    umbral_continua (float): Umbral mínimo del porcentaje de cardinalidad para considerar una variable como continua.

    Retorna:
    pd.DataFrame: Un DataFrame con dos columnas:
        - nombre_variable: Nombre de la columna en el DataFrame original.
        - tipo_sugerido: Tipo sugerido para la variable ("Binaria", "Categórica", "Numerica Continua", "Numerica Discreta").
    """
    unique_counts = df.nunique()  # Número de valores únicos por columna
    cardinality_percentage = (unique_counts / len(df)) * 100  # Porcentaje de cardinalidad

    resultados = []

    for col in df.columns:
        cardinalidad = unique_counts[col]
        porcentaje_cardinalidad = cardinality_percentage[col]

        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = "Numerica Continua"
            else:
                tipo_sugerido = "Numerica Discreta"

        resultados.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo_sugerido
        })

    return pd.DataFrame(resultados)  # DataFrame con la clasificación

# Ejemplo de uso
if __name__ == "__main__":
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "a", "c", "c"],
        "C": [1.5, 2.5, 3.5, 4.5, 5.5],
        "D": [1, 1, 1, 1, 0]
    }
    example_df = pd.DataFrame(data)

    # Usar tipifica_variables
    result_tipifica = tipifica_variables(example_df, umbral_categoria=3, umbral_continua=70.0)
    print("Clasificación de Variables:\n", result_tipifica)