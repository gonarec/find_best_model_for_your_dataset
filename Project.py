import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
import streamlit as st
import base64
from PIL import Image
from function_def import replace_outliers_with_median, replace_outliers_with_mean, remove_outliers,restore_function_corr, check_for_outliers
from function_def import  plot_boxplots, plot_boxplots_comparision, plot_bar_chart_df, plot_result,classificator_evo,encode_and_show_mapping


image_left = Image.open("ROB.jpeg")
image_right = Image.open("ROB.jpeg")

# Disposizione delle colonne
col1, col2, col3 = st.columns([1, 4, 1])

# Colonna sinistra con l'immagine
with col1:
    st.image(image_left, use_column_width=True)

# Colonna centrale con il titolo
with col2:
    st.markdown(
    "<h1 style='text-align: center; color: blue;'>FIND THE BEST MODEL TO CLASSIFY YOUR DATASET</h1>", 
    unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: left; color: orange;'>This tool product the accuracy of Random Forest, SVM, Bayes Gaussian and Decision Tree algorithms.<br> The accuracy depends on yor dataset, you can improve the accuracy using the following cleaning methods. </p>", unsafe_allow_html=True)
# Colonna destra con l'immagine
with col3:
    st.image(image_right, use_column_width=True)

st.write("")
st.write("")
st.markdown("<p style='text-align: left; color: yellow;'>INSERT YOUR DATABASE:</p>", unsafe_allow_html=True)
st.write("")

# Caricamento del file CSV e salvataggio del percorso in una variabile
file = st.file_uploader("Select a CSV file:")

# Elaborazione del file se è stato caricato
if file is not None:
    df = pd.read_csv(file)
    st.markdown("<p style='color: yellow;'>THIS IS YOUR DATASET:</p>", unsafe_allow_html=True)
    new_df = df.copy()
    st.dataframe(new_df)
    st.write(new_df.shape) 
    feature_to_classify = st.text_input("Which feature do you want classify?",None)
    if feature_to_classify is not None:
        new_df=encode_and_show_mapping(new_df)
        corr_matrix = new_df.corr().abs()
        corr_index = np.where((np.triu(corr_matrix, k=1) > 0.20))
        corr_features = pd.DataFrame({
            'Feature1': corr_matrix.columns[corr_index[0]], #Feature1
            'Feature2': corr_matrix.columns[corr_index[1]], #Feature2
            'Correlazione': corr_matrix.values[corr_index]
        })
    

 
    # Creiamo il plot del heatmap
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.title('Heatmap della matrice di correlazione')
    #st.pyplot(plt)

        with st.expander("DATA CLEANING - REMOVE OF NAN VALUE"):        
            if new_df.isnull().values.any():
                st.markdown("<p style='color: yellow;'>There are Nan values in your Dataset.</p>", unsafe_allow_html=True)
                option = st.radio("Do you want to remove the Nan values?",  ("No","Yes"))
                
                if option == "Yes":
                    st.markdown("<p style='color: yellow;'>You chose to remove Nan values.</p>", unsafe_allow_html=True)
                    st.write("Now choose between the options how do you want remove the Nan values. ")
                    cleaning_option = st.selectbox("Select cleaning function:", (None,"Drop raw with NaN", "Fill NaN with the mean of the column", 
                                                        "Fill NaN with the median fo the column", "Fill NaN with the mean of correlated data",
                                                        "Fill NaN with the meadian of correlated data"))
                    
                    if cleaning_option == "Drop raw with NaN":
                        new_df = new_df.dropna()
                        st.write("NaN values have been dropped.")
                    elif cleaning_option == "Fill NaN with the mean of the column":
                        new_df=new_df.fillna(new_df.mean())
                        st.write("NaN values have been filled.")
                    elif cleaning_option == "Fill NaN with the median fo the column":
                        new_df=new_df.fillna(new_df.median())
                        st.write("NaN values have been filled.")
                    elif cleaning_option == "Fill NaN with the mean of correlated data":
                        new_df=restore_function_corr(corr_features,0.10,new_df,'mean')
                        st.write("NaN values have been filled.")
                    elif cleaning_option == "Fill NaN with the meadian of correlated data":
                        new_df=restore_function_corr(corr_features,0.10,new_df,'median')
                        st.write("NaN values have been filled.")                                     
                    
                    if cleaning_option is not None:
                        st.markdown("<p style='color: yellow;'>This is your new dataset.</p>", unsafe_allow_html=True)
                        st.dataframe(new_df) 
                        st.write("Size of Dataset after cleaning:", new_df.shape)  

                        # Adding download button for cleaned DataFrame
                        csv = new_df.to_csv()
                        # Download button
                        st.download_button(label="Download the new CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

                elif option == "No":
                    st.markdown("<p style='color: yellow;'>If you dont remove the Nan value the perfomance of the classification can be compromised.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: yellow;'>The dataset is clean there are not Nan values inside.</p>", unsafe_allow_html=True)
        
        with st.expander("DATA CLEANING - OUTLINER"):
            if check_for_outliers==True:
                option1 = st.radio("Do you want to show the outliners?",  ("No","Yes"))
                if option1=="Yes":
                    st.markdown("<p style='color: yellow;'>BOXPLOT FEATURES: </p>", unsafe_allow_html=True)
                    plt.figure(figsize=(14, 10))
                    st.pyplot(plot_boxplots(new_df,feature_to_classify))
                    st.markdown("<p style='color: yellow;'>These are the boxplots of your dataset.</p>", unsafe_allow_html=True)
                    option = st.radio("Do you want to manage the outliner?",  ("No", "Yes"))
                    
                    if option == "Yes":
                            old_data=new_df.copy()
                            st.write("Now choose between the options how do you want remove the outliner. ")
                            cleaning_option = st.selectbox("Select cleaning function:", (None,"Replace with mean", "Replace with median", "Remove outliners"))
                            
                            if cleaning_option == "Replace with mean":
                                new_df=replace_outliers_with_mean(new_df,feature_to_classify)
                                st.write("NaN values have been dropped.")
                            elif cleaning_option == "Replace with median":
                                new_df=replace_outliers_with_median(new_df,feature_to_classify)
                            elif cleaning_option == "Remove outliners":
                                new_df=remove_outliers(new_df,feature_to_classify)
                            
                            if cleaning_option is not None:
                                st.markdown("<p style='color: yellow;'>THESE ARE YOUR NEW DATA:</p>", unsafe_allow_html=True)
                                plt.figure(figsize=(14, 10))
                                st.pyplot(plot_boxplots_comparision(old_data,new_df))
                                st.write("Size of Dataset after cleaning:", new_df.shape)  

                                csv = new_df.to_csv()
                                st.download_button(label="Download the new CSV", data=csv, file_name="new_data.csv", mime="text/csv")

                    if option == "No":
                        st.markdown("<p style='color: yellow;'>If you dont remove the outliners the perfomance of the classification can be compromised.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: yellow;'>You havent outliners in your dataset.</p>", unsafe_allow_html=True)

        with st.expander("CLASSIFICATION:"):
            result=pd.DataFrame()
            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider')
            if (feature_to_classify is not None and test_size != 0.0) :
                loading_message = st.empty()
                loading_message.info("Loading...")
                result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                loading_message.empty()
                plt.figure(figsize=(14, 10))
                st.pyplot(plot_bar_chart_df(result[test_size]))
            
                opt = st.radio("Do you confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key='opt')
                if opt == "Yes, show me the results":
                    st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                    st.dataframe(result)
                else:
                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider1')
                    if (feature_to_classify is not None and test_size != 0.0) :
                        loading_message = st.empty()
                        loading_message.info("Loading...")
                        result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                        loading_message.empty()
                        plt.figure(figsize=(14, 10))
                        st.pyplot(plot_bar_chart_df(result[test_size]))
                        opt = st.radio("Do you want confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key='opt1')
                        if opt == "Yes, show me the results":
                            st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                            st.dataframe(result)
                            plt.figure(figsize=(14, 10))
                            st.pyplot(plot_result(result))
                        else:
                            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider2')
                            if (feature_to_classify is not None and test_size != 0.0) :
                                loading_message = st.empty()
                                loading_message.info("Loading...")
                                result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                                loading_message.empty()
                                plt.figure(figsize=(14, 10))
                                st.pyplot(plot_bar_chart_df(result[test_size]))
                                opt = st.radio("Do you want confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key=f'opt2')
                                if opt == "Yes, show me the results":
                                    st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                                    st.dataframe(result)
                                    plt.figure(figsize=(14, 10))
                                    st.pyplot(plot_result(result))
                                    csv=result.to_csv()
                                    b64 = base64.b64encode(csv.encode()).decode()  # Codifica in base64
                                    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click here for the CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                else:
                                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider3')
                                    if (feature_to_classify is not None and test_size != 0.0) :
                                        loading_message = st.empty()
                                        loading_message.info("Loading...")
                                        result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                                        loading_message.empty()
                                        plt.figure(figsize=(14, 10))
                                        st.pyplot(plot_bar_chart_df(result[test_size]))
                                        opt = st.radio("Do you want confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key='opt3')
                                        if opt == "Yes, show me the results":
                                            st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                                            st.dataframe(result)
                                            plt.figure(figsize=(14, 10))
                                            st.pyplot(plot_result(result))
                                            csv=result.to_csv()
                                            b64 = base64.b64encode(csv.encode()).decode()  # Codifica in base64
                                            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click to download the result in CSV</a>'
                                            st.markdown(href, unsafe_allow_html=True)
                                        else:
                                            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider4')
                                            if (feature_to_classify is not None and test_size != 0.0) :
                                                loading_message = st.empty()
                                                loading_message.info("Loading...")
                                                result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                                                loading_message.empty()
                                                plt.figure(figsize=(14, 10))
                                                st.pyplot(plot_bar_chart_df(result[test_size]))
                                                opt = st.radio("Do you want confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key='opt4')
                                                if opt == "Yes, show me the results":
                                                    st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                                                    st.dataframe(result)
                                                    plt.figure(figsize=(14, 10))
                                                    st.pyplot(plot_result(result))
                                                    csv=result.to_csv()
                                                    b64 = base64.b64encode(csv.encode()).decode()  # Codifica in base64
                                                    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click to download the result in CSV</a>'
                                                    st.markdown(href, unsafe_allow_html=True)
                                                else:
                                                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider5')
                                                    if (feature_to_classify is not None and test_size != 0.0) :
                                                        loading_message = st.empty()
                                                        loading_message.info("Loading...")
                                                        result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                                                        loading_message.empty()
                                                        plt.figure(figsize=(14, 10))
                                                        st.pyplot(plot_bar_chart_df(result[test_size]))
                                                        opt = st.radio("Do you want confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key='opt5')
                                                        if opt == "Yes, show me the results":
                                                            st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
                                                            st.dataframe(result)
                                                            plt.figure(figsize=(14, 10))
                                                            st.pyplot(plot_result(result))
                                                            csv=result.to_csv()
                                                            b64 = base64.b64encode(csv.encode()).decode()  # Codifica in base64
                                                            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click to download the result in CSV</a>'
                                                            st.markdown(href, unsafe_allow_html=True)
                                                        else:
                                                            st.markdown("<p style='color: yellow;'>YOU CAN CONFRONT MAXIMUM SIX TESTSIZE VALUE </p>", unsafe_allow_html=True)
                                                            plt.figure(figsize=(14, 10))
                                                            st.pyplot(plot_result(result))
                                                            csv = result.to_csv()
                                                            b64 = base64.b64encode(csv.encode()).decode()  # Codifica in base64
                                                            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Click to download the result in CSV</a>'
                                                            st.markdown(href, unsafe_allow_html=True)
        
                       

