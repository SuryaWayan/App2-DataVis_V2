import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Summary",
    page_icon="📶",
    layout="wide"
)

data = st.session_state.get('data')
summary_data_1 = st.session_state.get('summary_data_1')
summary_data_2 = st.session_state.get('summary_data_2')
summary_data_3 = st.session_state.get('summary_data_3')
summary_data_4 = st.session_state.get('summary_data_4')
summary_data_5 = st.session_state.get('summary_data_5')
summary_data_6 = st.session_state.get('summary_data_6')
summary_data_7 = st.session_state.get('summary_data_7')
summary_data_8 = st.session_state.get('summary_data_8')
summary_data_9 = st.session_state.get('summary_data_9')
summary_data_10 = st.session_state.get('summary_data_10')
summary_data_11 = st.session_state.get('summary_data_11')
summary_data_12 = st.session_state.get('summary_data_12')
summary_data_13 = st.session_state.get('summary_data_13')
summary_data_14 = st.session_state.get('summary_data_14')
summary_data_15 = st.session_state.get('summary_data_15')
summary_data_16 = st.session_state.get('summary_data_16')
summary_data_17 = st.session_state.get('summary_data_17')
summary_data_18 = st.session_state.get('summary_data_18')
summary_data_19 = st.session_state.get('summary_data_19')
summary_data_20 = st.session_state.get('summary_data_20')


if data is not None:
    summary_df_1 = pd.DataFrame(summary_data_1)
    summary_df_2 = pd.DataFrame(summary_data_2)
    summary_df_3 = pd.DataFrame(summary_data_3)
    summary_df_4 = pd.DataFrame(summary_data_4)
    summary_df_5 = pd.DataFrame(summary_data_5)
    summary_df_6 = pd.DataFrame(summary_data_6)
    summary_df_7 = pd.DataFrame(summary_data_7)
    summary_df_8 = pd.DataFrame(summary_data_8)
    summary_df_9 = pd.DataFrame(summary_data_9)
    summary_df_10 = pd.DataFrame(summary_data_10)
    summary_df_11 = pd.DataFrame(summary_data_11)
    summary_df_12 = pd.DataFrame(summary_data_12)
    summary_df_13 = pd.DataFrame(summary_data_13)
    summary_df_14 = pd.DataFrame(summary_data_14)
    summary_df_15 = pd.DataFrame(summary_data_15)
    summary_df_16 = pd.DataFrame(summary_data_16)
    summary_df_17 = pd.DataFrame(summary_data_17)
    summary_df_18 = pd.DataFrame(summary_data_18)
    summary_df_19 = pd.DataFrame(summary_data_19)
    summary_df_20 = pd.DataFrame(summary_data_20)


    summary_df = pd.concat([summary_df_1, summary_df_2, summary_df_3, summary_df_4, summary_df_5, summary_df_6, summary_df_7, summary_df_8, summary_df_9, summary_df_10, summary_df_11, summary_df_12, summary_df_13, summary_df_14, summary_df_15, summary_df_16, summary_df_17, summary_df_18, summary_df_19, summary_df_20], ignore_index=True)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.warning("Please upload a CSV file first.")