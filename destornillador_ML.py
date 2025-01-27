#Carga de paquetes y módulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, spearmanr,pearsonr
import seaborn as sns


#1. Función describe_df
def describe_df(df):
    """
    Genera un resumen descriptivo de un DataFrame.
    """
    summary = {
        "type": df.dtypes,
        "null_percentage": df.isnull().mean() * 100,
        "unique_values": df.nunique(),
        "cardinality_percentage": (df.nunique() / len(df)) * 100
    }
    return pd.DataFrame(summary)

# Función tipifica_variables
def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las columnas de un DataFrame según su tipo sugerido.
    """
    unique_counts = df.nunique()
    cardinality_percentage = (unique_counts / len(df)) * 100

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
        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo_sugerido})
    return pd.DataFrame(resultados)


#2. Función: get_features_num_regression

def get_features_num_regression(dataframe, target_col, umbral_corr, pvalue=None):
    """
    Seleccionar columnas numéricas correlacionadas con la variable objetivo.
    Parámetros:
    - dataframe: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo.
    - umbral_corr: Umbral mínimo de correlación en valor absoluto.
    - pvalue: Valor p para pruebas de hipótesis (opcional).

    Devuelve:
    - Lista de columnas numéricas correlacionadas con la columna objetivo.
    """

    if target_col not in dataframe.columns or not np.issubdtype(dataframe[target_col].dtype, np.number):
        print("La columna objetivo no es válida o no es numérica continua.")
        return None

    correlaciones = [] 

    for columna in dataframe.select_dtypes(include=[np.number]).columns:
        if columna != target_col:  
            corr, p = pearsonr(dataframe[columna].dropna(), dataframe[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or p <= pvalue):
                correlaciones.append(columna)

    return correlaciones 

#3. Funcion: plot_features_num_regression
def plot_features_num_regression(dataframe, target_col, columns=[], umbral_corr=0, pvalue=None):
    """
    Generar pairplot de columnas numéricas correlacionadas con la variable objetivo.
    Parámetros:
    - dataframe: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo.
    - columns: Lista de columnas a evaluar (opcional).
    - umbral_corr: Umbral mínimo de correlación en valor absoluto (por defecto 0).
    - pvalue: Valor p para pruebas de hipótesis (opcional).

    Devuelve:
    - Lista de columnas numéricas seleccionadas para el gráfico.
    """
    if not columns:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    columnas_validas = get_features_num_regression(dataframe, target_col, umbral_corr, pvalue)

    columnas_a_graficar = [col for col in columns if col in columnas_validas]

    if len(columnas_a_graficar) > 5:
        for i in range(0, len(columnas_a_graficar), 5):
            sns.pairplot(dataframe, vars=[target_col] + columnas_a_graficar[i:i + 5])
            plt.show() 
    else:
        sns.pairplot(dataframe, vars=[target_col] + columnas_a_graficar)
        plt.show()  

    return columnas_a_graficar


#4. Funcion: get_features_cat_regression

def get_features_cat_regression(df, target_col, pvalue):

    """
    Identifica columnas categóricas en un DataFrame cuya relación estadística con la columna objetivo es significativa.
    Aplica ANOVA si la columna objetivo es continua y correlación de Spearman si es discreta.

    Argumentos:
    - df (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Nombre de la columna objetivo (debe ser numérica).
    - pvalue (float): Nivel de significancia estadística.

    Retorna:
    - dict: Diccionario con las columnas categóricas significativas y sus respectivos p-valores.
    """

    # Validaciones iniciales
    #Comprobar DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("La entrada no es un DataFrame válido.")
    #Comprobar si existe la columna target y que es una variable numérica.
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna objetivo '{target_col}' no es numérica.")
    #Comprobar que el pvalue intruducido sea una valor entre 0 y 1
    if not (0 < pvalue < 1):
        raise ValueError("El valor de 'pvalue' debe estar entre 0 y 1.")

    # Identificar columnas categóricas (cardinalidad <10), sino devuelve {} vacía.
    cat_columns = [col for col in df.columns if col != target_col and (pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 10)]
    if not cat_columns:
        print("No se encontraron columnas categóricas en el DataFrame.")
        return {}
    
    # Identificar si target_col es continua o discreta
    is_continuous = df[target_col].nunique() > 10

    # Preparar el diccionario de columnas categóricas significativas
    significant_features = {}

    # Iterar sobre cada columna categórica
    for col in cat_columns:
        # Filtrar filas con valores nulos
        df_valid = df.dropna(subset=[col, target_col])
        if is_continuous:
            # ANOVA: Comparar medias entre grupos
            groups = [df_valid[df_valid[col] == category][target_col] for category in df_valid[col].unique()]
            if len(groups) > 1:  # Asegurarse de que haya al menos 2 grupos
                f_stat, p_val = f_oneway(*groups)
            else:
                p_val = 1  # No es posible realizar ANOVA con un solo grupo
        else:
            # Spearman: Correlación para variables discretas
            unique_categories = pd.get_dummies(df_valid[col], drop_first=True)
            p_vals = []
            for category in unique_categories.columns:
                _, p_val = spearmanr(unique_categories[category], df_valid[target_col])
                p_vals.append(p_val)
            p_val = min(p_vals)  # Usar el p-value más bajo

        # Si la relación es estadísticamente significativa, agregar al diccionario
        if p_val < pvalue:
            significant_features[col] = p_val

    return significant_features

#5. Funcion: plot_features_cat_regression

def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Visualiza la relación entre variables categóricas y una variable objetivo numérica mediante histogramas agrupados.

    Argumentos:
    dataframe (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo. Debe ser numérica.
    columns (list): Lista de strings de columnas categóricas a analizar. Por defecto es una lista vacía.
    pvalue (float): Por defecto es 0.05.
    with_individual_plot (bool): Por defecto es False, y no se generan gráficos. Si es True, genera gráficos individuales para cada variable categórica seleccionada.

    Devuelve:
    list: Lista de columnas categóricas que tienen una relación significativa con la columna objetivo.
    """
    # Verificación
    if not isinstance(dataframe, pd.DataFrame): #comprobar si un objeto pertenece a una clase o tipo específico
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas.")

    if not isinstance(target_col, str) or target_col == "":
        raise ValueError("El argumento 'target_col' debe ser una cadena no vacía.")

    if target_col not in dataframe.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en el DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        raise ValueError(f"La columna objetivo '{target_col}' debe ser de tipo numérico.")

    if not isinstance(columns, list):
        raise ValueError("El argumento 'columns' debe ser una lista.")

    if not columns:
        columns = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()

    significant_columns = []

    for col in columns:
        if col not in dataframe.columns:
            continue

        if not pd.api.types.is_categorical_dtype(dataframe[col]) and not pd.api.types.is_object_dtype(dataframe[col]):
            continue #Verifica si la columna col es categorica o de tipo object

        # Prueba ANOVA para verificar relación significativa
        groups = [dataframe[dataframe[col] == category][target_col] for category in dataframe[col].dropna().unique()]
        if len(groups) > 1:
            stat, p = f_oneway(*groups) #descompone la lista groups en elementos individuales, pasando cada grupo como un argumento separado.
            if p < pvalue:
                significant_columns.append(col)

                # Generar gráfico si se solicita
                if with_individual_plot:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=col, y=target_col, data=dataframe)
                    plt.title(f"Relación entre {col} y {target_col} (p-value=0.05)")
                    plt.xticks(rotation=45)
                    plt.show()

    return significant_columns