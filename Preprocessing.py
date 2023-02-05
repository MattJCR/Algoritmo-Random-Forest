import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# python pip install -U scikit-learn pandas numpy
class CleanData:
    _le = preprocessing.LabelEncoder()
    '''
    CleanData: Clase para limpiar DataFrame y prepararlos para el entrenamiento de un modelo.
    Autor: Mateo Jesus Cabello Rodriguez
    '''
    def clean(df,drop_columns=None,delete_null_values=False,fill_null_values=False):
        '''
        Prepara un DataFrame para ser usado en predicciones.
        df: 'Requiere de un objecto DataFrame'
        drop_columns: 'Listado de columnas que se quieren eliminar del DataFrame'
        delete_null_values: 'Boolean que indica si se borran filas/tuplas con valores nulos'
        fill_null_values: 'Boolean que indica si se rellena con el valor de la media las filas/tuplas que tengan valores nulos'
        '''
        if(df.__class__ != pd.DataFrame):
            print("Se esperaba un DataFrame:",df.__class__)
            return None
        if(fill_null_values):
            df = CleanData.fill_numeric_null_rows(df)
        if(delete_null_values):
            df = CleanData.delete_null_rows(df)
        df = CleanData.clean_columns(df)
        df = CleanData.convert_types(df)
        if(drop_columns != None and type(drop_columns) == list):
            df = CleanData.drop_colums(df,drop_columns)
        return df
    def delete_null_rows(df):
        """
        Borra las filas/tuplas del dataframe que contengan algun valor nulo.
        df: 'Requiere de un objecto DataFrame'
        """
        return df.copy().dropna(inplace = True)
    def fill_numeric_null_rows(df):
        """
        Rellena con la media las filas/tuplas del dataframe que contengan algun valor nulo.
        df: 'Requiere de un objecto DataFrame'
        """
        df = df.copy()
        numeric_columns = CleanData.get_numeric_columns(df)
        for column in numeric_columns:
            df.fillna(int(df[column].mean()), inplace = True)
        return df
        
    def drop_colums(df,drop_columns):
        """
        Borra las columnas del dataframe.
        df: 'Requiere de un objecto DataFrame'
        drop_columns: 'Listado de columnas que se quieren eliminar del DataFrame'
        """
        for column in drop_columns:
            df.drop(column.lower(), axis=1, inplace = True)
        return df
    def clean_columns(df):
        '''
        Transforma el nombre de las columnas a lowercase.
        df: 'Requiere de un objecto DataFrame'
        '''
        df.columns = map(str.lower, df.columns)
        return df
    def convert_types(df):
        '''
        Convierte las columnas del DataFrame en sus tipos correspondientes.
        Los tipos son numeric, date y str.
        df: 'Requiere de un objecto DataFrame'
        '''
        df = CleanData.convert_numeric_types(df)
        df = CleanData.convert_date_types(df)
        df = CleanData.convert_str_types(df)
        return df
    def convert_numeric_types(df):
        '''
        Convierte las columnas del DataFrame de tipo numerico en int o float.
        df: 'Requiere de un objecto DataFrame'
        '''
        numeric_columns = CleanData.get_numeric_columns(df)
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column],errors='coerce')
        return df
    def convert_date_types(df):
        '''
        Convierte las columnas del DataFrame de tipo fecha en datetime.
        df: 'Requiere de un objecto DataFrame'
        '''
        date_columns = CleanData.get_date_columns(df)
        for column in date_columns:
            df[column] = pd.to_datetime(df[column],errors='coerce')
        return df
    def convert_str_types(df):
        '''
        Convierte las columnas del DataFrame de tipo cadena en str.
        Limpia los espacios de las cadenas y los reemplaza por _ .
        df: 'Requiere de un objecto DataFrame'
        '''
        df= df.applymap(lambda s:s.lower().strip().replace(' ','_') if type(s) == str else s)
        return df
    def get_categorical_columns(df):
        '''
        Devuelve las columnas que son de tipo categórico.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.dtypes[df.dtypes == 'object'].to_dict().keys()
    def get_numeric_columns(df):
        '''
        Devuelve las columnas que son de tipo numerico.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.select_dtypes(include=np.number).columns.tolist()
    def get_date_columns(df):
        '''
        Devuelve las columnas que son de tipo datetime.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.select_dtypes(include=np.datetime64).columns.tolist()
    def transform_categorical_to_numeric(df):
        '''
        Convierte las columnas del DataFrame que son categóricas en int.
        df: 'Requiere de un objecto DataFrame'
        '''
        categorical = CleanData.get_categorical_columns(df)
        for column in categorical:
            df[column] = CleanData._le.fit_transform(df[column].values)
        return df
    
    def z_score_outliers(df,column,standard_deviation=3):
        '''
        Devuelve los outliers y z_scores de una columna de un dataframe.
        df: 'Requiere de un objecto DataFrame'
        column: 'Nombre de columna para sacar los outliers'
        standard_deviation: 'Valor de desviación para marcar como outlier'
        '''

        mean = df[column].mean()
        std = df[column].std()

        z_scores = (df[column] - mean) / std
        outliers = df[(np.abs(z_scores) > standard_deviation)]
        return outliers, z_scores
    
    def remove_outliers(df,outliers):
        '''
        Borra los outliers del dataframe.
        df: 'Requiere de un objecto DataFrame'
        outliers: 'DataFrame con los outliers detectados'
        '''
        return df[~df.index.isin(outliers.index)]
    
class ViewData:
    def box_plot_representation(df,outliers,column):
        '''
        Muestra una representación gráfica de Box Plot con los outliers detectados.
        df: 'Requiere de un objecto DataFrame'
        outliers: 'DataFrame con los outliers detectados'
        column: 'Nombre de columna para sacar los outliers'
        '''
        fig, ax = plt.subplots()
        ax.boxplot(df[column], showfliers=False)
        for i in outliers.index:
            ax.scatter(1, outliers.loc[i, column], marker='o', color='red')
        plt.show()

    def histogram_representation(df,outliers,column):
        '''
        Muestra una representación gráfica de Histogram con los outliers detectados.
        df: 'Requiere de un objecto DataFrame'
        outliers: 'DataFrame con los outliers detectados'
        column: 'Nombre de columna para sacar los outliers'
        '''
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=50)

        for i in outliers.index:
            ax.scatter(outliers.loc[i, column], 0, marker='o', color='red')
        plt.show()

    def violin_plot_representation(df,outliers,column):
        '''
        Muestra una representación gráfica de Violín Plot con los outliers detectados.
        df: 'Requiere de un objecto DataFrame'
        outliers: 'DataFrame con los outliers detectados'
        column: 'Nombre de columna para sacar los outliers'
        '''
        sns.violinplot(df[column])

        for i in outliers.index:
            plt.scatter(outliers.loc[i, column], 0, marker='o', color='red')
        plt.show()

    def probability_density_function_representation(outliers,z_scores):
        '''
        Muestra una representación gráfica de Probability Density Function con los outliers detectados.
        df: 'Requiere de un objecto DataFrame'
        outliers: 'DataFrame con los outliers detectados'
        column: 'Nombre de columna para sacar los outliers'
        '''
        sns.distplot(z_scores, kde=True, rug=True)

        for i in outliers.index:
            plt.scatter(z_scores.loc[i], 0, marker='o', color='red')
        plt.show()

    def show_heatmap(df):
        '''
        Calcular la correlación entre las variables del DataFrame y Muestra el mapa de calor de la correlación.
        df: 'Requiere de un objecto DataFrame'
        '''
        corr = df.corr()

        plt.figure(figsize=(17, 15))
        sns.heatmap(corr,square=True,annot=True,linewidths=1.5)
        plt.show()