import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go  # No need to import plotly.graph_objects again
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Upload CSV",
    page_icon="📂",
    layout="wide"
)

# CSV file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV data
    data = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
    # Store data in session state
    st.session_state.data = data
