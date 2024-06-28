import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps

from sklearn.preprocessing import LabelEncoder
import streamlit as st

def check_for_outliers(dataframe, threshold=1.5):
    # Calcola il valore del terzo quartile (Q3) e il primo quartile (Q1) per ciascuna feature
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)

    # Calcola l'intervallo interquartile (IQR) per ciascuna feature
    IQR = Q3 - Q1

    # Determina i limiti per il rilevamento degli outlier
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Controlla se ci sono valori che cadono al di fuori degli intervalli per ciascuna feature
    outlier_present = ((dataframe < lower_bound) | (dataframe > upper_bound)).any().any()

    # Restituisci True se ci sono outlier, altrimenti False
    return outlier_present

def encode_and_show_mapping(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])

            # Stampa il mapping delle categorie solo se la colonna contiene stringhe
            st.write("Convert the string in number for the column:", column,"")
            for i, category in enumerate(label_encoder.classes_):
                st.write(f"Sobstitute {category} with -> {i}")

    return df
def replace_outliers_with_median(df,feature):
    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != feature:
            q1 = df[col].quantile(0.25)  # Calcola il primo quartile
            q3 = df[col].quantile(0.75)  # Calcola il terzo quartile
            iqr = q3 - q1  # Calcola l'interquartile range (IQR)

            # Calcola i limiti per gli outlier
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Sostituisci gli outlier con la mediana
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].apply(lambda x: median if x < lower_bound or x > upper_bound else x)

    return df_clean

def replace_outliers_with_mean(df,feature):
    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != feature:
            q1 = df[col].quantile(0.25)  # Calcola il primo quartile
            q3 = df[col].quantile(0.75)  # Calcola il terzo quartile
            iqr = q3 - q1  # Calcola l'interquartile range (IQR)

            # Calcola i limiti per gli outlier
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Sostituisci gli outlier con la mediana
            mean = df_clean[col].mean()
            df_clean[col] = df_clean[col].apply(lambda x: mean if x < lower_bound or x > upper_bound else x)

    return df_clean

def remove_outliers(df,feature):

    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != feature:
            q1 = df[col].quantile(0.25)  # Calcola il primo quartile
            q3 = df[col].quantile(0.75)  # Calcola il terzo quartile
            iqr = q3 - q1  # Calcola l'interquartile range (IQR)

            # Calcola i limiti per gli outlier
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Rimuovi gli outlier
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

def plot_boxplots(dataframe, feature):
    # Seleziona solo le colonne con più di due valori unici
    relevant_columns = [col for col in dataframe.columns if dataframe[col].nunique() > 2]

    num_plots = len(relevant_columns)
    cols_per_row = 4 

    # Calcola il numero di righe necessarie
    num_rows = (num_plots - 1) // cols_per_row + 1

    # Crea il layout dei subplot
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

    # Flatten l'array di assi se è multidimensionale
    axes = axes.flatten()

    # Itera sulle colonne del DataFrame
    for i, col in enumerate(relevant_columns):
        # Seleziona l'asse corrente
        ax = axes[i]

        # Disegna il boxplot per la feature corrente
        sns.boxplot(x=feature, y=col, data=dataframe, ax=ax, palette='coolwarm')

        # Imposta il titolo del boxplot
        ax.set_title(f'Boxplot di {col}')

        # Ruota le etichette sull'asse x per una migliore leggibilità
        ax.tick_params(axis='x', rotation=45)

        # Imposta le etichette sull'asse y con precisione a due cifre decimali
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Rimuovi gli assi vuoti se ce ne sono
    for ax in axes[num_plots:]:
        ax.remove()

    # Imposta il layout dei subplot
    plt.tight_layout()

    return fig


def plot_boxplots_comparision(dataframe_1, dataframe_clean_1):
    dataframe = dataframe_1.copy()
    dataframe_clean = dataframe_clean_1.copy()
    
    # Filtra solo le colonne con più di due valori unici
    relevant_columns = [col for col in dataframe.columns if dataframe[col].nunique() > 2]
    
    num_plots = len(relevant_columns)  # Numero di colonne nel DataFrame con più di due valori unici
    cols_per_row = 4

    # Calcola il numero di righe necessarie
    num_rows = (num_plots - 1) // cols_per_row + 1

    # Crea il layout dei subplot
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

    # Flatten l'array di assi se è multidimensionale
    axes = axes.flatten()

    # Concatena i DataFrame relevant e relevant_clean, aggiungendo una colonna 'Dataset' per distinguere tra i due
    dataframe['Dataset'] = 'Prima'
    dataframe_clean['Dataset'] = 'Dopo'

    # Unisci i DataFrame
    combined_df = pd.concat([dataframe, dataframe_clean], ignore_index=True)

    # Itera su ogni feature con più di due valori unici
    for i, col in enumerate(relevant_columns):
        ax = axes[i]
        sns.boxplot(x='Dataset', y=col, data=combined_df, hue='Dataset', palette=["blue", "orange"], ax=ax)
        ax.set_title(f'Rimozione Outlier di {col}')
            
    # Rimuovi gli assi vuoti se ce ne sono
    for ax in axes[num_plots:]:
        ax.remove()

    # Imposta il layout dei subplot
    plt.tight_layout()

    return fig


def plot_bar_chart_df(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classifiers = df.index.tolist()

    bar_width = 0.2
    index = np.arange(len(classifiers))

    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        bar_positions = index + i * bar_width
        ax.bar(bar_positions, values, bar_width, label=metric)

        for j, v in enumerate(values):
            ax.text(bar_positions[j], v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Metrics by Classifier')
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(classifiers)
    ax.legend()
        # Impostazione dei limiti dell'asse y
    max_value = df[metrics].values.max() * 1.2  
    ax.set_ylim(0, max_value)

    plt.tight_layout()
    return fig

def plot_result(res):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classifiers = res.index.unique().tolist()
    test_sizes = res.columns.levels[0].tolist()

    fig, axs = plt.subplots(len(test_sizes), 1, figsize=(15, 5 * len(test_sizes)))

    if len(test_sizes) == 1:
        axs = [axs]

    bar_width = 0.2
    for i, test_size in enumerate(test_sizes):
        ax = axs[i]
        df = res[test_size]
        index = np.arange(len(classifiers))

        for j, metric in enumerate(metrics):
            values = df[metric].tolist()
            bar_positions = index + j * bar_width
            ax.bar(bar_positions, values, bar_width, label=metric)

            for k, v in enumerate(values):
                ax.text(bar_positions[k], v + 0.01, f"{v:.2f}", ha='center', va='bottom')

        ax.set_title(f'Test Size = {test_size}')
        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Performance')
        ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(classifiers)
        ax.legend()

            # Impostazione dei limiti dell'asse y
        max_value = df[metrics].values.max() * 1.2  # Aumenta l'asse y del 20% rispetto al valore massimo
        ax.set_ylim(0, max_value)

    plt.tight_layout()
    return fig

def classificator_evo(dataset, classifier, testsize):
    metrics_dict = {}

    x_data = dataset.drop(columns=[classifier])
    y_data = dataset[classifier]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=testsize, random_state=10)

    # Neural Network (MLPClassifier)
    mlp_classifier = MLPClassifier(max_iter=500, random_state=42)
    mlp_classifier.fit(x_train, y_train)
    y_pred = mlp_classifier.predict(x_test)

    metrics_dict['NN'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
        'recall': round(recall_score(y_test, y_pred, average='weighted'), 3),
        'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 3)
    }

    # Support Vector Machine (SVC)
    svm_classifier = SVC(kernel='rbf', C=4.0, gamma='scale', random_state=42)
    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)

    metrics_dict['SVM'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
        'recall': round(recall_score(y_test, y_pred, average='weighted'), 3),
        'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 3)
    }

    # Decision Tree Classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)
    tree_classifier.fit(x_train, y_train)
    y_pred = tree_classifier.predict(x_test)

    metrics_dict['Tree'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
        'recall': round(recall_score(y_test, y_pred, average='weighted'), 3),
        'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 3)
    }

    # Naive Bayes Classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(x_train, y_train)
    y_pred = naive_bayes_classifier.predict(x_test)

    metrics_dict['Bayes'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
        'recall': round(recall_score(y_test, y_pred, average='weighted'), 3),
        'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 3)
    }

    # Convert metrics_dict to DataFrame and add testsize
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df['testsize'] = testsize

    return df

#def classificator_evo (dataset, classifier, testsize):
    accuracy_dict={}
    x_data=dataset.drop(columns=[classifier])
    y_data=dataset.loc[:,classifier]
    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=testsize,random_state=10)
    
   ## RandomForestClassifier 
    #rf_model = RandomForestClassifier(n_estimators=100, random_state=20)
   # rf_model.fit(x_train, y_train)

    #y_pred = rf_model.predict(x_test)

    # Valutazione delle prestazioni del modello
   # accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello RandomForestClassifier: %.3f" %accuracy)
  #  accuracy_dict['RandomForest']=round(accuracy,3)

    mlp_classifier = MLPClassifier(max_iter=500, random_state=42)

    mlp_classifier.fit(x_train, y_train)
    y_pred = mlp_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['NN']=round(accuracy,3)

    svm_classifier = SVC(kernel='rbf', C=4.0, gamma='scale', random_state=42)

    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['SVM']=round(accuracy,3)

    # Crea il modello di regressione logistica con solo relevant
    #logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, random_state=42)

    #logistic_regression.fit(x_train, y_train)
    #y_pred = logistic_regression.predict(x_test)

    #accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello di regressione logistica: %.3f" %accuracy)
    #accuracy_dict['Regression']=accuracy

    # DecisionTreeClassifier con tutte le feature
    tree_classifier = DecisionTreeClassifier(random_state=42)

    tree_classifier.fit(x_train, y_train)
    y_pred = tree_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del classificatore ad albero decisionale: %.3f" %accuracy)
    accuracy_dict['Tree']=round(accuracy,3)


    # Classificatore naive bayes con relevant feature
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(x_train, y_train)

    y_pred = naive_bayes_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del classificatore Naive Bayes: %.3f" %accuracy)
    accuracy_dict['Bayes']=round(accuracy,3)

    return accuracy_dict


def restore_function_corr(corr_features, delta, dataframe, mean):
    labels_with_nan = dataframe.columns[dataframe.isna().any()].tolist()
    for label in labels_with_nan:
        #Sostituisce le label in Feature2 da confrontare e le mette in Feature1 di relevant row
        relevant_rows = corr_features[(corr_features['Feature1'] == label) | (corr_features['Feature2'] == label)]
        swap_mask = relevant_rows['Feature2'] == label
        relevant_rows.loc[swap_mask, ['Feature1', 'Feature2']] = relevant_rows.loc[swap_mask, ['Feature2', 'Feature1']].values
        relevant_rows = relevant_rows.sort_values(by='Correlazione', ascending=False)
        relevant_rows.reset_index(drop=True, inplace=True)

        #dove non ce una seconda feature ne crea una fittizia di Nan
        try:
            second_corr_label_value = relevant_rows.iloc[1]['Feature2']
        except IndexError:
            second_corr_label_value = np.nan   #add a nan row
            relevant_rows.loc[len(relevant_rows)] = np.nan

        df_miss = dataframe[dataframe[label].isna()]
        for count, value in enumerate(df_miss[relevant_rows.iloc[0]['Feature2']]):
            indx=df_miss[relevant_rows.iloc[0]['Feature2']].index[count]
            if pd.isna(value):
                if pd.isna(relevant_rows.iloc[1]['Feature2']):
                    if mean=='mean':
                        dataframe.loc[indx,label] = round(dataframe[label].mean(skipna=True),2)
                    else:
                        dataframe.loc[indx,label] = round(dataframe[label].median(skipna=True),2)
                else:
                    if mean=='mean': 
                        new_value=dataframe.loc[indx,relevant_rows.iloc[1]['Feature2']]
                        correlated=dataframe[
                            (dataframe[relevant_rows.iloc[1]['Feature2']] >= new_value*(1-delta)) &
                            (dataframe[relevant_rows.iloc[1]['Feature2']] <= new_value*(1+delta))]
                        dataframe.loc[indx,label] = round(correlated[label].mean(skipna=True),2)
                    else:
                        new_value=dataframe.loc[indx,relevant_rows.iloc[1]['Feature2']]
                        correlated=dataframe[
                            (dataframe[relevant_rows.iloc[1]['Feature2']] >= new_value*(1-delta)) &
                            (dataframe[relevant_rows.iloc[1]['Feature2']] <= new_value*(1+delta))]
                        dataframe.loc[indx,label] = round(correlated[label].median(skipna=True),2)

            else:
                if mean == 'mean':
                    correlated=dataframe[
                        (dataframe[relevant_rows.iloc[0]['Feature2']] >= value*(1-delta)) &
                        (dataframe[relevant_rows.iloc[0]['Feature2']] <= value*(1+delta))]
                    dataframe.loc[indx,label] = round(correlated[label].mean(skipna=True), 2)
                else:
                    correlated=dataframe[
                        (dataframe[relevant_rows.iloc[0]['Feature2']] >= value*(1-delta)) &
                        (dataframe[relevant_rows.iloc[0]['Feature2']] <= value*(1+delta))]
                    dataframe.loc[indx,label] = round(correlated[label].median(skipna=True), 2)               

    return dataframe