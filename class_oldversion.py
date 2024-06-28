
###OLD VERSION

with st.expander("CLASSIFICATION:"):
            result=pd.DataFrame()
            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider')
            if (feature_to_classify is not None and test_size != 0.0) :  #1
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
                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider1') #2
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
                            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider2') #3
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
                                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider3') #4
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
                                            test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider4') #5
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
                                                    test_size = st.slider("Select the test size dimension:", min_value=0.0, max_value=0.9, step=0.05, value=0.0, key='slider5') #6
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
        
                       

