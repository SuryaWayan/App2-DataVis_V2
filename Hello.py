import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go  # No need to import plotly.graph_objects again
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide"
)

col_header1, col_header2, col_header3 = st.columns([3, 4, 1])

with col_header3:
    st.write("Antero Eng Tool - by SA")

st.write("# Holla! Welcome to Antero Eng Tool! 👋")

st.sidebar.success("Please Select the above menu to navigate between pages.")

st.markdown(
    """
    This is a web app used specifically for Data Visualization and Simple Analysis.
"""
)
st.markdown(
    """
    **👈 Please select a menu from the sidebar** 
"""
)

