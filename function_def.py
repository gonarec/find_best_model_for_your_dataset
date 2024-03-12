import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps

def replace_outliers_with_median(df):
    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != 'quality':
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

def replace_outliers_with_mean(df):
    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != 'quality':
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

def remove_outliers(df):

    df_clean = df.copy()  # Copia il DataFrame originale per non modificarlo direttamente

    for col in df.columns:
        if col != 'quality':
            q1 = df[col].quantile(0.25)  # Calcola il primo quartile
            q3 = df[col].quantile(0.75)  # Calcola il terzo quartile
            iqr = q3 - q1  # Calcola l'interquartile range (IQR)

            # Calcola i limiti per gli outlier
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Rimuovi gli outlier
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

def new_quality_value(df):
    
    new_df = df.copy()
    # Mappa i valori della colonna 'quality' 
    quality_mapping = {3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2}
    new_df['quality'] = new_df['quality'].replace(quality_mapping)
    
    return new_df

def classificator (dataset, classifier):
    accuracy_dict={}
    x_data=dataset.drop(columns=[classifier])
    y_data=dataset.loc[:,classifier]
    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.10,random_state=10)

    # RandomForestClassifier 
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=20)
    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    # Valutazione delle prestazioni del modello
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello RandomForestClassifier: %.3f" %accuracy)
    accuracy_dict['RandomForest']=round(accuracy,3)

    # Classificatore SVM con solo relevant feature
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del classificatore SVM: %.3f" %accuracy)
    accuracy_dict['SVM']=round(accuracy,3)

    # Crea il modello di regressione logistica con solo relevant
    logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, random_state=42)

    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello di regressione logistica: %.3f" %accuracy)
    accuracy_dict['Regression']=accuracy

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

def plot_boxplots(dataframe):
    num_plots = len(dataframe.columns) - 1  
    cols_per_row = 4 

    # Calcola il numero di righe necessarie
    num_rows = (num_plots - 1) // cols_per_row + 1

    # Crea il layout dei subplot
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

    # Flatten l'array di assi se è multidimensionale
    axes = axes.flatten()

    # Itera sulle colonne del DataFrame escludendo "quality"
    for i, col in enumerate(dataframe.drop(columns='quality')):
        # Seleziona l'asse corrente
        ax = axes[i]

        # Disegna il boxplot per la feature corrente con "quality" sulle x
        sns.boxplot(x='quality', y=col, data=dataframe, ax=ax, palette='coolwarm', hue='quality', legend=False)

        # Imposta il titolo del boxplot
        ax.set_title(f'Boxplot di {col}')

        # Ruota le etichette sull'asse x per una migliore leggibilità
        ax.tick_params(axis='x', rotation=45)

        # Imposta le etichette sull'asse y con precisione a tre cifre decimali
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Rimuovi gli assi vuoti se ce ne sono
    for ax in axes[num_plots:]:
        ax.remove()

    # Imposta il layout dei subplot
    plt.tight_layout()

    return fig

def plot_boxplots_comparision(dataframe_1, dataframe_clean_1):
    dataframe=dataframe_1.copy()
    dataframe_clean=dataframe_clean_1.copy()
    num_plots = len(dataframe.columns) - 1 # Numero di colonne nel DataFrame escludendo "quality" e Dataset
    cols_per_row = 4 

    # Calcola il numero di righe necessarie
    num_rows = (num_plots - 1) // cols_per_row + 1

    # Crea il layout dei subplot
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))

    # Flatten l'array di assi se è multidimensionale
    axes = axes.flatten()

    # Concateniamo i DataFrame relevant e relevant_clean, aggiungendo una colonna 'Dataset' per distinguere tra i due
    dataframe['Dataset'] = 'Prima'
    dataframe_clean['Dataset'] = 'Dopo'

    # Uniamo i DataFrame
    combined_df = pd.concat([dataframe, dataframe_clean])

    # Iteriamo su ogni feature
    for i, col in enumerate(dataframe.columns):
        if col != 'quality' and col != 'Dataset':
            ax = axes[i]
            sns.boxplot(x='Dataset', y=col, data=combined_df, hue='Dataset', palette=["blue", "orange"], ax=ax)
            ax.set_title(f'Rimozione Outlier di {col}')
            
    # Rimuovi gli assi vuoti se ce ne sono
    for ax in axes[num_plots:]:
        ax.remove()

    # Imposta il layout dei subplot
    plt.tight_layout()
    plt.show()

def plot_bar_chart_df(df):
    if type(df) == pd.DataFrame:
        keys = df.index.tolist()  # Ottieni gli indici del DataFrame come chiavi
        values = df.iloc[:, 0].tolist()  # Ottieni i valori dalla prima colonna del DataFrame
    elif type(df) == pd.Series:
        keys = df.index.tolist()
        values = df.tolist()

    colors = plt.cm.RdBu(np.array(values) / max(values))  # Utilizza la mappa di colori "RdBu" in base ai valori massimi

    # Aggiungi valori sulle barplot
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    plt.bar(keys, values, color=colors)
    plt.xlabel('Algoritmi')
    plt.ylabel('Performance')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(values) * 1.2)  # Estendi l'asse y del 10%
    plt.show()

def plot_result(res):
    num_cols = len(res.columns)
    num_rows = (num_cols + 1) // 2 

    # Creazione del grafico a barre
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))

    # Itera sul DataFrame e crea i subplot
    for i, (col_name, col_data) in enumerate(res.items()):
        row = i // 2
        col = i % 2
        keys = res.index
        values = col_data
        colors = plt.cm.RdBu(np.array(values) / max(values))
        
        for j, v in enumerate(values):
            axs[row, col].text(j, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        
        axs[row, col].bar(keys, values, color=colors)
        axs[row, col].set_ylabel('Valori')
        axs[row, col].set_title('Dataset name='+col_name)
        axs[row, col].set_xticks(keys)
        axs[row, col].set_xticklabels(keys, rotation=45, ha='right')
        axs[row, col].set_ylim(0, max(values) * 1.2)

    # Rimuovi i subplot non utilizzati
    for i in range(num_cols, num_rows*2):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()

def classificator_evo (dataset, classifier, testsize):
    accuracy_dict={}
    x_data=dataset.drop(columns=[classifier])
    y_data=dataset.loc[:,classifier]
    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=testsize,random_state=10)

    accuracy_dict['Size']= dataset.shape[0]
    
    # RandomForestClassifier 
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=20)
    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    # Valutazione delle prestazioni del modello
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello RandomForestClassifier: %.3f" %accuracy)
    accuracy_dict['RandomForest']=round(accuracy,3)

    # Classificatore SVM con solo relevant feature
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del classificatore SVM: %.3f" %accuracy)
    accuracy_dict['SVM']=round(accuracy,3)

    # Crea il modello di regressione logistica con solo relevant
    logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, random_state=42)

    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuratezza del modello di regressione logistica: %.3f" %accuracy)
    accuracy_dict['Regression']=accuracy

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

def classification_evo(dataframe, testsize):

    data=dataframe.copy()
    data_clean_median=replace_outliers_with_median(data)
    data_clean_mean=replace_outliers_with_mean(data)
    data_clean_remove=remove_outliers(data)

    relevant=dataframe.drop(columns=['fixed_acidity','residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH'])
    relevant_clean_median=replace_outliers_with_median(relevant)
    relevant_clean_mean=replace_outliers_with_mean(relevant)
    relevant_clean_remove=remove_outliers(relevant)


    result=pd.DataFrame()

    result['Data'+'_'+str(testsize)]=classificator_evo(data,'quality',testsize)

    result['Relevant'+'_'+str(testsize)]=classificator_evo(relevant,'quality', testsize)

    result['Data_Clean_Median'+'_'+str(testsize)]=classificator_evo(data_clean_median,'quality', testsize)

    result['Relevant_Clean_Median'+'_'+str(testsize)]=classificator_evo(relevant_clean_median, 'quality', testsize)

    result['Data_Clean_Mean'+'_'+str(testsize)]=classificator_evo(data_clean_mean,'quality', testsize)

    result['Relevant_Clean_Mean'+'_'+str(testsize)]=classificator_evo(relevant_clean_mean, 'quality', testsize)

    result['Data_Remove'+'_'+str(testsize)]=classificator_evo(data_clean_remove,'quality', testsize)

    result['Relevant_Remove'+'_'+str(testsize)]=classificator_evo(relevant_clean_remove, 'quality', testsize)

    return result

def plot_result_evo(res):
    num=res.iloc[0]
    result=res.drop(index= ['Size'])
    num_cols = len(result.columns)
    num_rows = (num_cols + 1) // 2 

    # Creazione del grafico a barre
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))

    # Itera sul DataFrame e crea i subplot
    for i, (col_name, col_data) in enumerate(result.items()):
        row = i // 2
        col = i % 2
        keys = result.index
        values = col_data
        colors = plt.cm.RdBu(np.array(values) / max(values))
        
        for j, v in enumerate(values):
            axs[row, col].text(j, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        
        axs[row, col].bar(keys, values, color=colors)
        axs[row, col].set_ylabel('Valori')
        axs[row, col].set_title('Dataset name='+col_name+'   Dataset size='+str(num.iloc[i]))
        axs[row, col].set_xticks(keys)
        axs[row, col].set_xticklabels(keys, rotation=45, ha='right')
        axs[row, col].set_ylim(0, max(values) * 1.2)

    # Rimuovi i subplot non utilizzati
    for i in range(num_cols, num_rows*2):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()

def plot_bar_chart_df_evo(df):
  
    keys = df.index.tolist()  
    values = df.iloc[:, 0].tolist()  
    x_labels = [f"{key}: {df.iloc[i, 1]}" for i, key in enumerate(keys)]  
   

    colors = plt.cm.RdBu(np.array(values) / max(values)) 

    # Aggiungi valori sulle barplot
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    plt.bar(keys, values, color=colors)
    plt.xlabel('Algoritmi')
    plt.ylabel('Performance')
    plt.xticks(range(len(keys)), x_labels, rotation=45, ha='right')
    plt.ylim(0, max(values) * 1.2)
    plt.show()

def trova_max(df):
    # Trova il massimo in ogni riga
    max_values = df.max(axis=1)

    # Trova l'indice del massimo in ogni riga
    max_indices = df.idxmax(axis=1)

    max_df = pd.DataFrame({'Max_Value': max_values, 'Index': max_indices})
    return max_df

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