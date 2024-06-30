with st.expander("CLASSIFICATION:"):
    result=pd.DataFrame()
    flag = False
    test_size1 = 2.0
    key_counter = 0  # Inizializza un contatore per generare ID univoci per i widget
    flag1=True
    feature_to_classify = st.text_input("Which feature do you want classify?", key='feature')
    while not flag:
        if flag1==True:
            key_counter += 1  # Incrementa il contatore per ottenere un nuovo ID univoco
            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key=f'slider_{key_counter}')
            flag1=False
            if (feature_to_classify is not None) and (test_size != 0.0) and (test_size != test_size1):
                loading_message = st.empty()
                loading_message.info("Loading...")
                result[test_size] = classificator_evo(new_df, feature_to_classify, test_size)
                loading_message.empty()
                plt.figure(figsize=(14, 10))
                flag1=True
                st.pyplot(plot_bar_chart_df(result[test_size]))
                test_size1 = test_size
            
                option2 = st.radio("Do you confront your different testsize?",  ("No, i want get another testsize","Yes, show me the results"), key=f'option_{key_counter}')
                if option2 == "Yes, show me the results":
                    flag = True
                    break

    st.markdown("<p style='color: yellow;'>THESE ARE YOUR TEST:</p>", unsafe_allow_html=True)
    st.dataframe(result)