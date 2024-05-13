import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Setting",
    page_icon="🔧",
    layout="wide"
)

data = st.session_state.get('data')
if data is not None:
    height = st.session_state.get('height', 400)
    def handle_height():
        if st.session_state.handle_height:
            st.session_state.height = st.session_state.handle_height
    height = st.number_input("Chart Height", min_value=100, max_value=1000, value=height, on_change=handle_height, key='handle_height')
    st.session_state.height = height
else:
    st.warning("Please upload a CSV file first.")