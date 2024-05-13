import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Data Introduction",
    page_icon="📑",
    layout="wide"
)

# Retrieve data from session state
data = st.session_state.get('data')
selected_columns = st.session_state.get('selected_columns', [])

if data is not None:
    col_dataoverview1, col_dataoverview2, col_dataoverview3 = st.columns([0.5, 0.5, 4])

    with col_dataoverview1:
        st.write(f"**Total Rows:** {data.shape[0]}")
    with col_dataoverview2:
        st.write(f"**Total Columns:** {data.shape[1]}")

    columns = data.columns.tolist()

    # Interactive column selection
    def handle():
        if st.session_state.handle1:
            st.session_state.selected_columns = st.session_state.handle1
    selected_columns = st.multiselect("Please choose the columns you wish to display in table format, or skip this section if you prefer not to generate a table.", columns, selected_columns, on_change=handle, key='handle1')
    
    # Interactive table generation
    if selected_columns:
        num_rows = st.number_input("Number of rows to display", min_value=1, value=10)
        st.dataframe(data[selected_columns].head(num_rows))
    
    st.session_state.selected_columns = selected_columns

else:
    st.warning("Please upload a CSV file first.")
