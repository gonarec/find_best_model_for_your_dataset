# Import delle librerie necessarie
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import base64
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from function_def import replace_outliers,restore_function_corr, check_for_outliers
from function_def import  plot_boxplots, plot_boxplots_comparision, plot_bar_chart_df, plot_result,classificator_evo,encode_and_show_mapping

# IMPOSTAZIONI PAGINA INIZIALE
st.set_page_config(page_title="Dataset Classifier", layout="wide")
image_left = Image.open("ROB.jpeg")
image_right = Image.open("ROB.jpeg")
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.image(image_left, use_column_width=True)

with col2:
    st.markdown("<h1 style='text-align: center; color: blue;'>FIND THE BEST MODEL TO CLASSIFY YOUR DATASET</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: left; color: orange;'>This tool produces the accuracy, precision, recall, and F1 score of various classifiers (Neural Network, SVM, Decision Tree, Naive Bayes) on the inserted dataset. There are also data cleaning functions to enhance your model.</p>", unsafe_allow_html=True)

with col3:
    st.image(image_right, use_column_width=True)

st.write("")
st.write("")
st.markdown("<p style='text-align: left; color: yellow;'>INSERT YOUR DATABASE:</p>", unsafe_allow_html=True)
st.write("Only use datasets with numerical or binary categorical values.")

# CARICAMENTO DATASET
file = st.file_uploader("Select a CSV file:")

try:
    if file is not None:
        df = pd.read_csv(file)
        st.markdown("<p style='color: yellow;'>THIS IS YOUR DATASET:</p>", unsafe_allow_html=True)
        st.dataframe(df)
        st.write("Dataset Shape:", df.shape)

        feature_to_classify = st.text_input("Which feature do you want to classify?", None)

        if feature_to_classify is not None:
            df = encode_and_show_mapping(df,feature_to_classify)
            
            
            ##DATA CLEANING
            with st.expander("DATA CLEANING - REMOVE OF NAN VALUE"):
                if df.isnull().values.any():
                    st.markdown("<p style='color: yellow;'>There are Nan values in your Dataset.</p>", unsafe_allow_html=True)
                    option = st.radio("Do you want to remove the Nan values?", ("No", "Yes"))

                    if option == "Yes":

                        #Calcolo feature correlate, creazione nuovo df correlazione
                        corr_matrix = df.corr().abs()
                        corr_index = np.where(np.triu(corr_matrix, k=1) > 0.20)
                        corr_features = pd.DataFrame({
                            'Feature1': corr_matrix.columns[corr_index[0]],
                            'Feature2': corr_matrix.columns[corr_index[1]],
                            'Correlation': corr_matrix.values[corr_index] })
                        
                        st.markdown("<p style='color: yellow;'>You chose to remove Nan values.</p>", unsafe_allow_html=True)
                        st.write("Now choose how you want to remove the Nan values.")

                        cleaning_option = st.selectbox("Select cleaning function:", ("Drop rows with NaN", "Fill NaN with mean of the column",
                                                                                  "Fill NaN with median of the column",
                                                                                  "Fill NaN with mean of correlated data",
                                                                                  "Fill NaN with median of correlated data"))
                        if cleaning_option == "Drop rows with NaN":
                            df = df.dropna()
                            st.write("NaN values have been dropped.")

                        elif cleaning_option == "Fill NaN with mean of the column":
                            df = df.fillna(df.mean())
                            st.write("NaN values have been filled with mean of the column.")

                        elif cleaning_option == "Fill NaN with median of the column":
                            df = df.fillna(df.median())
                            st.write("NaN values have been filled with median of the column.")

                        elif cleaning_option == "Fill NaN with mean of correlated data":
                            df = restore_function_corr(corr_features, 0.10, df, 'mean')
                            st.write("NaN values have been filled with mean of correlated data.")

                        elif cleaning_option == "Fill NaN with median of correlated data":
                            df = restore_function_corr(corr_features, 0.10, df, 'median')
                            st.write("NaN values have been filled with median of correlated data.")

                        if cleaning_option is not None:
                            st.markdown("<p style='color: yellow;'>This is your new dataset:</p>", unsafe_allow_html=True)
                            st.dataframe(df)
                            st.write("Size of Dataset after cleaning:", df.shape)

                            csv = df.to_csv(index=False)
                            st.download_button(label="Download the new CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

                    elif option == "No":
                        st.markdown("<p style='color: yellow;'>If you don't remove the Nan values, the performance of the classification can be compromised.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: yellow;'>The dataset is clean; there are not Nan values.</p>", unsafe_allow_html=True)

            with st.expander("DATA CLEANING - OUTLIERS"):
                if check_for_outliers(df):
                    old_df=df.copy()
                    option = st.radio("Do you want to show the outliers?", ("No", "Yes"))
                    if option == "Yes":
                        st.markdown("<p style='color: yellow;'>BOXPLOT FEATURES: </p>", unsafe_allow_html=True)
                        plt.figure(figsize=(14, 10))
                        st.pyplot(plot_boxplots(df, feature_to_classify))
                        st.markdown("<p style='color: yellow;'>These are the boxplots of your dataset.</p>", unsafe_allow_html=True)
                        option = st.radio("Do you want to manage the outliers?", ("No", "Yes"))

                        if option == "Yes":
                            st.write("Now choose how you want to manage the outliers.")

                            cleaning_option = st.selectbox("Select cleaning function:", ("Replace with mean","Replace with median","Remove outliers"))
                            if cleaning_option == "Replace with mean":
                                df = replace_outliers(df, feature_to_classify,'mean')
                                st.write("Outliers have been replaced with mean.")

                            elif cleaning_option == "Replace with median":
                                df = replace_outliers(df, feature_to_classify,'median')
                                st.write("Outliers have been replaced with median.")

                            elif cleaning_option == "Remove outliers":
                                df = replace_outliers(df, feature_to_classify,'remove')
                                st.write("Outliers have been removed.")

                            if cleaning_option is not None:
                                st.markdown("<p style='color: yellow;'>This is your new dataset after handling outliers:</p>", unsafe_allow_html=True)
                                plt.figure(figsize=(14, 10))
                                st.pyplot(plot_boxplots_comparision(old_df,df,feature_to_classify))
                                st.write("Size of Dataset after cleaning:", df.shape)

                                csv = df.to_csv(index=False)
                                st.download_button(label="Download the new CSV", data=csv, file_name="cleaned_data_with_outliers.csv", mime="text/csv")

                        elif option == "No":
                            st.markdown("<p style='color: yellow;'>If you don't manage the outliers, the performance of the classification can be compromised.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: yellow;'>You don't have outliers in your dataset.</p>", unsafe_allow_html=True)

            with st.expander("MODEL SETTINGS"):

                st.markdown("<p style='color: yellow;'>SVM:</p>", unsafe_allow_html=True)

                svm_c= st.slider("Select C:", min_value=0.0, max_value=20.0, step=0.5, value=1.0, key='sliderC')
                svm_kernel = st.selectbox("Select SVM kernel:", ("rbf","linear","poly","sigmoid","precomputed" ))
                if svm_kernel == "rbf":
                    svm_kernel='rbf'
                elif svm_kernel == "linear":
                    svm_kernel='linear'
                elif svm_kernel == "poly":
                    svm_kernel="poly"
                elif svm_kernel == "sigmoid":
                    svm_kernel='sigmoid'
                elif svm_kernel == "precomputed":
                    svm_kernel='precomputed'                


                st.markdown("<p style='color: yellow;'>NEURAL NETWORK:</p>", unsafe_allow_html=True)

                n_layers = st.slider('Number of hidden layer:', min_value=1, max_value=10, value=1)
                hidden_layer_sizes = []
                for i in range(n_layers):
                    neurons = st.slider(f'Numero of neurons in the layer {i+1}', min_value=1, max_value=200, value=100)
                    hidden_layer_sizes.append(neurons)

                activation_nn = st.selectbox("Select NN activation function:", ("identity","logistic","tanh","relu" ))
                if activation_nn == "identity":
                    activation_nn='identity'
                elif activation_nn == "logistic":
                    activation_nn='logistic'
                elif activation_nn == "tanh":
                    activation_nn='tanh'
                elif activation_nn == "relu":
                    activation_nn='relu'
                

                st.markdown("<p style='color: yellow;'>BAYES:</p>", unsafe_allow_html=True)

                st.write("For the Bayes Gaussian Classificator are used the default parameters of sklearn.")


                st.markdown("<p style='color: yellow;'>DECISION TREE:</p>", unsafe_allow_html=True)
                criterion_tree = st.selectbox('Select Decision Tree criterion:', ('gini', 'entropy'), index=0)
                splitter_tree = st.selectbox('Select splitter strategy for Decision Tree:', ('best', 'random'), index=0)


            ## PARTE DI CLASSIFICAZIONE
            with st.expander("CLASSIFICATION:"):
                result = pd.DataFrame()
                test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider')

                counter = 0

                while feature_to_classify is not None and test_size != 0.0:
                    loading_message = st.empty()
                    loading_message.info("Loading...")

                    metrics_df = classificator_evo(df, feature_to_classify, test_size, svm_c, svm_kernel, hidden_layer_sizes, activation_nn, criterion_tree, splitter_tree)

                    # Aggiungi multi-indice per gestire le diverse dimensioni del test set
                    metrics_df.columns = pd.MultiIndex.from_product([[f'Test Size {test_size}'], metrics_df.columns])

                    if result.empty:
                        result = metrics_df
                    else:
                        result = pd.concat([result, metrics_df], axis=1)

                    loading_message.empty()

                    plt.figure(figsize=(14, 10))
                    st.pyplot(plot_bar_chart_df(metrics_df.xs(f'Test Size {test_size}', axis=1, level=0)))

                    slider_key = f'slider_{counter}'
                    counter += 1
                    opt_key = f'opt_{counter}'
                    counter += 1

                    opt = st.radio("Do you want to confront your different test sizes?", ("No, I want to get another test size", "Yes, show me the results"), key=opt_key)

                    if opt == "Yes, show me the results":
                        st.markdown("<p style='color: yellow;'>THESE ARE YOUR TESTS:</p>", unsafe_allow_html=True)
                        performance_columns = result.columns.get_level_values(1)
                        performance_columns = performance_columns[~performance_columns.str.startswith('Test Size')]

                        # Filtra il DataFrame result per mantenere solo le colonne di performance
                        result_to_display = result.loc[:, (slice(None), performance_columns)]

                        st.dataframe(result_to_display)

                        if not result.empty:
                            plt.figure(figsize=(14, 10))
                            st.pyplot(plot_result(result))
                            csv = result.to_csv()
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click to download the result in CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='color: red;'>No results to display.</p>", unsafe_allow_html=True)

                        break
                    else:
                        test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key=slider_key)

                if not result.empty:
                    plt.figure(figsize=(14, 10))
except Exception as e:
    st.error(f"An error occurred: {str(e)}")