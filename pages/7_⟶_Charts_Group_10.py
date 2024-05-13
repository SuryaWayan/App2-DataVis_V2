import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Data 10",
    page_icon="📊",
    layout="wide"
)

# Show charts?
Chart_10 = st.session_state.get('Chart_10', False)
def handle_Chart_10():
    if st.session_state.handle_Chart_10:
        st.session_state.Chart_10 = st.session_state.handle_Chart_10
Chart_10 = st.checkbox("Click to Show charts", Chart_10, on_change=handle_Chart_10, key='handle_Chart_10')
st.session_state.Chart_10 = Chart_10

if Chart_10:
    # Retrieve data from session state
    data = st.session_state.get('data')
    summary_data_10 = []
    num_charts_n = st.session_state.get('num_charts_n', 1)  # Default value is 1
    start_x_all_10 = st.session_state.get('start_x_all_10', None)
    end_x_all_10 = st.session_state.get('end_x_all_10', None)
    tunnel_10 = st.session_state.get('tunnel_10', False)
    threshold_10 = st.session_state.get('threshold_10', 0.00000)
    C_10_option = st.session_state.get('C_10_option', None)
    C_10 = st.session_state.get('C_10', 0.00000)
    C_10_column = st.session_state.get('C_10_column', "Row Number")
    B_10 = st.session_state.get('B_10', 0.00000)
    D_10 = st.session_state.get('D_10', 0.00000)
    A_10 = st.session_state.get('A_10', [])
    height = st.session_state.get('height', 400)


    # Adding a new column "Number/Duration"
    if data is not None:
        data['Row Number'] = range(len(data))
        columns = data.columns.tolist()

        # Configure charts
        def handle_num_charts_n():
            if st.session_state.handle_num_charts_n:
                st.session_state.num_charts_n = st.session_state.handle_num_charts_n
        num_charts_n = st.number_input("Number of charts to generate", min_value=1, max_value=10, value=num_charts_n, on_change=handle_num_charts_n, key='handle_num_charts_n')
        st.session_state.num_charts_n = num_charts_n
        st.write("")
        st.write("")
        
        # Input fields for x-axis range for all charts
        x_min = data['Row Number'].min()
        x_max = data['Row Number'].max()

        if start_x_all_10 is None or end_x_all_10 is None:
            start_x_all_10 = x_min
            end_x_all_10 = x_max

        st.write("Select X-axis range for all charts:")
        col_x_range_start_10, col_x_range_end_10, col_remain_10 = st.columns([0.6, 0.6, 5])
        with col_x_range_start_10:
            def handle_start_x_all_10():
                if st.session_state.handle_start_x_all_10:
                    st.session_state.start_x_all_10 = st.session_state.handle_start_x_all_10
            start_x_all_10 = st.number_input("Enter X-axis start", min_value=x_min, max_value=x_max, value=start_x_all_10, on_change=handle_start_x_all_10, key='handle_start_x_all_10')
        with col_x_range_end_10:
            def handle_end_x_all_10():
                if st.session_state.handle_end_x_all_10:
                    st.session_state.end_x_all_10 = st.session_state.handle_end_x_all_10
            end_x_all_10 = st.number_input("Enter X-axis end", min_value=x_min, max_value=x_max, value=end_x_all_10, on_change=handle_end_x_all_10, key='handle_end_x_all_10')
        st.session_state.start_x_all_10 = start_x_all_10
        st.session_state.end_x_all_10 = end_x_all_10
        st.write("")
        st.write("")

        # Function to calculate upper and lower limits for specified columns
        def calculate_limits(column_data, threshold, D_10, B_10, C_10_option, data):
            lower_limit = upper_limit = np.nan
            if pd.api.types.is_numeric_dtype(column_data):
                lower_limit = column_data.copy()
                upper_limit = column_data.copy()
                
                if C_10_option == "User Input":
                    C_10_value = C_10
                elif C_10_option == "From Existing Variables":
                    if "C_10_column" in data.columns:
                        C_10_value = data["C_10_column"]
                    else:
                        st.error("Please select a column for C_10_column.")
                        return np.nan, np.nan
                else:
                    st.error("Invalid option selected for C_10_option.")
                    return np.nan, np.nan
                
                lower_limit[column_data <= threshold] -= D_10 + (B_10 * C_10_value / 100)
                upper_limit[column_data <= threshold] += D_10 + (B_10 * C_10_value / 100)
                upper_limit[column_data > threshold] += B_10 * C_10_value / 100
                lower_limit[column_data > threshold] -= B_10 * C_10_value / 100
                
            return lower_limit, upper_limit

        # Adding Tunnel. Check if X and Y values are provided and the chart type is not Bar
        def handle_tunnel_10():
            if st.session_state.handle_tunnel_10:
                st.session_state.tunnel_10 = st.session_state.handle_tunnel_10
        tunnel_10 = st.checkbox("Add tunnel - lower & upper limit of variables", tunnel_10, on_change=handle_tunnel_10, key='handle_tunnel_10')
        st.session_state.tunnel_10 = tunnel_10
            
        if tunnel_10:
            col_notef_10, col_noteg_10 = st.columns([0.05,3.5])
            with col_noteg_10:
                st.write("     If A > threshold, tunnel = A ± (B% x C)")
                st.write("     If A ≤ threshold, tunnel = A ± (B% x C) ± D")

            col_noteh_10, col_notea_10, col_noteb_10, col_notec_10, col_noted_10, col_notee_10 = st.columns([0.08,1,1,1,1,1])

            with col_notea_10:
                def handle_threshold_10():
                        if st.session_state.handle_threshold_10:
                            st.session_state.threshold_10 = st.session_state.handle_threshold_10
                threshold_10 = st.number_input("Enter threshold value", min_value=0.00000, max_value=999999999.00000, value=threshold_10, on_change=handle_threshold_10, key='handle_threshold_10')
                st.session_state.threshold_10 = threshold_10

            with col_noteb_10:
                def handle_C_10_option():
                        if st.session_state.handle_C_10_option:
                            st.session_state.C_10_option = st.session_state.handle_C_10_option
                C_10_option = st.selectbox("Select option for C:", ["User Input", "From Existing Variables"], index=0 if C_10_option == "User Input" else 1, on_change=handle_C_10_option, key='handle_C_10_option')
                st.session_state.C_10_option = C_10_option
                if C_10_option == "User Input":
                    C_10 = st.number_input("Enter C", min_value=0.00000, max_value=999999999.00000, value=C_10)
                    st.session_state.C_10 = C_10

                elif C_10_option == "From Existing Variables":
                    #C_10_column = st.session_state.get('C_10_column', "Row Number")
                    numeric_columns = list(data.select_dtypes(include=np.number).columns)  
                    default_index = numeric_columns.index(C_10_column)
                    def handle_C_10_column():
                        if st.session_state.handle_C_10_column:
                            st.session_state.C_10_column = st.session_state.handle_C_10_column
                    C_10_column = st.selectbox("Select variables for C:", numeric_columns, index=default_index, on_change=handle_C_10_column, key='handle_C_10_column')
                    st.session_state['C_10_column'] = C_10_column
                    data["C_10_column"] = data[C_10_column]

            with col_notec_10:
                def handle_B_10():
                        if st.session_state.handle_B_10:
                            st.session_state.B_10 = st.session_state.handle_B_10
                B_10 = st.number_input("Enter B in %", min_value=0.00000, max_value=100.00000, value=B_10, on_change=handle_B_10, key='handle_B_10')
                st.session_state.B_10 = B_10

            with col_noted_10:
                def handle_D_10():
                        if st.session_state.handle_D_10:
                            st.session_state.D_10 = st.session_state.handle_D_10
                D_10 = st.number_input("Enter D", min_value=0.00000, max_value=999999999.00000, value=D_10, on_change=handle_D_10, key='handle_D_10')
                st.session_state.D_10 = D_10
                
            with col_notee_10:
                def handle_A_10():
                    if st.session_state.handle_A_10:
                        st.session_state.A_10 = st.session_state.handle_A_10
                A_10 = st.multiselect('Select A: variables to calculate limits for:', data.columns, A_10, on_change=handle_A_10, key='handle_A_10')    
                st.session_state.A_10 = A_10

            st.write("")
            st.write("")
                        
            for column in A_10:
                # Check if column exists in data
                if column in data.columns:
                    # Calculate upper and lower limits for the selected column
                    lower_limit, upper_limit = calculate_limits(data[column], threshold_10, D_10, B_10, C_10_option, data)
                    # Add upper and lower limits as new columns
                    data[f"{column}_lower_limit"] = lower_limit
                    data[f"{column}_upper_limit"] = upper_limit



        for i in range(num_charts_n):
            st.subheader(f"Chart {i+1}")

            col1_n, col_chart_n = st.columns([1, 3])

            with col1_n:
                chart_type_n = st.selectbox(f"Select chart type for Chart {i+1}", ["Line", "Bar", "Scatter"], key=f"chart_type_n_{i}")

                # Initialize warning message
                warning_message = st.empty()
                
                # Retrieve or initialize session state variables for this chart
                x_column_key_n = f"x_column_n_{i+1}"  
                x_column_n = st.session_state.get(x_column_key_n, "Row Number") 
                def handle_x_column_n(i):  
                    handle_key_x_10 = f'handle_x_columns_n_{i+1}'
                    if st.session_state.get(handle_key_x_10):
                        st.session_state[x_column_key_n] = st.session_state[handle_key_x_10]  
                x_column_n = st.selectbox(f"Select X-axis column for Chart {i+1}", columns, index=columns.index(x_column_n), on_change=lambda: handle_x_column_n(i), key=f'handle_x_columns_n_{i+1}')  # Modified
                st.session_state[x_column_key_n] = x_column_n 

                if i == 0:
                    i = 0
                    y_columns_key_n_m0 = f'y_columns_n_{i}'  
                    y_columns_n = st.session_state.get(y_columns_key_n_m0, [])  
                    def handle_y_columns_n_m0():
                        handle_key_n_m0 = f'handle_y_columns_n_m0'
                        if st.session_state.get(handle_key_n_m0):
                            st.session_state[y_columns_key_n_m0] = st.session_state[handle_key_n_m0]  
                    y_columns_n_m0 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m0, key=f'handle_y_columns_n_m0')  
                    st.session_state[y_columns_key_n_m0] = y_columns_n  
                
                elif i == 1:
                    i = 1
                    y_columns_key_n_m1 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m1, [])  
                    def handle_y_columns_n_m1():
                        handle_key_n_m1 = f'handle_y_columns_n_m1'
                        if st.session_state.get(handle_key_n_m1):
                            st.session_state[y_columns_key_n_m1] = st.session_state[handle_key_n_m1]  
                    y_columns_n_m1 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m1, key=f'handle_y_columns_n_m1')  
                    st.session_state[y_columns_key_n_m1] = y_columns_n  

                elif i == 2:
                    i = 2
                    y_columns_key_n_m2 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m2, [])  
                    def handle_y_columns_n_m2():
                        handle_key_n_m2 = f'handle_y_columns_n_m2'
                        if st.session_state.get(handle_key_n_m2):
                            st.session_state[y_columns_key_n_m2] = st.session_state[handle_key_n_m2]  
                    y_columns_n_m2 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m2, key=f'handle_y_columns_n_m2')  
                    st.session_state[y_columns_key_n_m2] = y_columns_n  

                elif i == 3:
                    i = 3
                    y_columns_key_n_m3 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m3, [])  
                    def handle_y_columns_n_m3():
                        handle_key_n_m3 = f'handle_y_columns_n_m3'
                        if st.session_state.get(handle_key_n_m3):
                            st.session_state[y_columns_key_n_m3] = st.session_state[handle_key_n_m3]  
                    y_columns_n_m3 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m3, key=f'handle_y_columns_n_m3')  
                    st.session_state[y_columns_key_n_m3] = y_columns_n  

                elif i == 4:
                    i = 4
                    y_columns_key_n_m4 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m4, [])  
                    def handle_y_columns_n_m4():
                        handle_key_n_m4 = f'handle_y_columns_n_m4'
                        if st.session_state.get(handle_key_n_m4):
                            st.session_state[y_columns_key_n_m4] = st.session_state[handle_key_n_m4]  
                    y_columns_n_m4 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m4, key=f'handle_y_columns_n_m4')  
                    st.session_state[y_columns_key_n_m4] = y_columns_n 

                elif i == 5:
                    i = 5
                    y_columns_key_n_m5 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m5, [])  
                    def handle_y_columns_n_m5():
                        handle_key_n_m5 = f'handle_y_columns_n_m5'
                        if st.session_state.get(handle_key_n_m5):
                            st.session_state[y_columns_key_n_m5] = st.session_state[handle_key_n_m5]  
                    y_columns_n_m5 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m5, key=f'handle_y_columns_n_m5')  
                    st.session_state[y_columns_key_n_m5] = y_columns_n  

                elif i == 6:
                    i = 6
                    y_columns_key_n_m6 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m6, [])  
                    def handle_y_columns_n_m6():
                        handle_key_n_m6 = f'handle_y_columns_n_m6'
                        if st.session_state.get(handle_key_n_m6):
                            st.session_state[y_columns_key_n_m6] = st.session_state[handle_key_n_m6]  
                    y_columns_n_m6 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m6, key=f'handle_y_columns_n_m6')  
                    st.session_state[y_columns_key_n_m6] = y_columns_n 

                elif i == 7:
                    i = 7
                    y_columns_key_n_m7 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m7, [])  
                    def handle_y_columns_n_m7():
                        handle_key_n_m7 = f'handle_y_columns_n_m7'
                        if st.session_state.get(handle_key_n_m7):
                            st.session_state[y_columns_key_n_m7] = st.session_state[handle_key_n_m7]  
                    y_columns_n_m7 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m7, key=f'handle_y_columns_n_m7')  
                    st.session_state[y_columns_key_n_m7] = y_columns_n 

                elif i == 8:
                    i = 8
                    y_columns_key_n_m8 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m8, [])  
                    def handle_y_columns_n_m8():
                        handle_key_n_m8 = f'handle_y_columns_n_m8'
                        if st.session_state.get(handle_key_n_m8):
                            st.session_state[y_columns_key_n_m8] = st.session_state[handle_key_n_m8]  
                    y_columns_n_m8 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m8, key=f'handle_y_columns_n_m8')  
                    st.session_state[y_columns_key_n_m8] = y_columns_n  

                elif i == 9:
                    i = 9
                    y_columns_key_n_m9 = f'y_columns_n_{i}' 
                    y_columns_n = st.session_state.get(y_columns_key_n_m9, [])  
                    def handle_y_columns_n_m9():
                        handle_key_n_m9 = f'handle_y_columns_n_m9'
                        if st.session_state.get(handle_key_n_m9):
                            st.session_state[y_columns_key_n_m9] = st.session_state[handle_key_n_m9]  
                    y_columns_n_m9 = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns, y_columns_n, on_change=handle_y_columns_n_m9, key=f'handle_y_columns_n_m9')  
                    st.session_state[y_columns_key_n_m9] = y_columns_n 


                # Check if both X and Y values are provided
                if len(x_column_n) > 0 and len(y_columns_n) > 0:
                    # If both X and Y values are provided, remove the warning message
                    warning_message.empty()
                else:
                # If either X or Y values are missing, display the warning message
                    warning_message.warning("Please input X and Y values with NUMBER format, not DATE or TEXT.")

                # Check if X and Y values are provided and the chart type is not Bar
                if chart_type_n != "Bar" and len(x_column_n) > 0 and len(y_columns_n) > 0:
                    trendline_n = st.checkbox(f"Add trendline for Chart {i+1}.", key=f"trendline_n_{i}")
                else:
                    trendline_n = False

                if trendline_n:
                    col_a, col_b = st.columns([1.5, 1.5])
                    with col_a:
                        trendline_type_n = st.selectbox(f"Select trendline type", ["Linear", "Average", "Polynomial"], key=f"trendline_type_n_{i}")
                    with col_b:
                        if trendline_type_n == "Polynomial":
                            degrees = {}
                            for y_column in y_columns_n:
                                degrees[y_column] = st.number_input(f"Degree for {y_column}", min_value=2, max_value=20, value=2, key=f"degree_{i}_{y_column}")
                st.subheader("")

            with col_chart_n:
                with st.container():
                    if x_column_n and y_columns_n:
                        filtered_data = data[(data['Row Number'] >= start_x_all_10) & (data['Row Number'] <= end_x_all_10)]

                        if chart_type_n == "Line":
                            fig = px.line(filtered_data, x=x_column_n, y=y_columns_n)
                        elif chart_type_n == "Bar":
                            fig = px.bar(filtered_data, x=x_column_n, y=y_columns_n)
                        else:
                            fig = px.scatter(filtered_data, x=x_column_n, y=y_columns_n)

                        if trendline_n:
                            for trace in fig.data:
                                if trace.name in y_columns_n:
                                    color = trace.line.color if hasattr(trace.line, 'color') else trace.marker.color
                                    if trendline_type_n == "Linear":
                                        slope, intercept, _, _, _ = linregress(filtered_data[x_column_n], filtered_data[trace.name])
                                        fig.add_shape(type="line", x0=start_x_all_10, y0=slope*start_x_all_10+intercept,
                                                    x1=end_x_all_10, y1=slope*end_x_all_10+intercept,
                                                    line=dict(color=color, width=2, dash="dash"))
                                        y_predicted = slope * filtered_data[x_column_n] + intercept
                                        r_squared = r2_score(filtered_data[trace.name], y_predicted)
                                    elif trendline_type_n == "Average":
                                        avg_y_value = filtered_data[trace.name].mean()
                                        fig.add_hline(y=avg_y_value, line_dash="dash",
                                                    annotation_text=f"Avg {trace.name} = {avg_y_value:.5f}",
                                                    annotation_position="bottom right",
                                                    line=dict(color=color)) 
                                    elif trendline_type_n == "Polynomial":
                                        degree = degrees[trace.name]
                                        coeffs = np.polyfit(filtered_data[x_column_n], filtered_data[trace.name], degree)
                                        poly_function = np.poly1d(coeffs)
                                        equation = " + ".join(f"{coeffs[i]:.8f} * x^{degree-i}" for i in range(degree+1))
                                        x_values = np.linspace(start_x_all_10, end_x_all_10, 100)
                                        y_values = poly_function(x_values)
                                        r_squared = r2_score(filtered_data[trace.name], poly_function(filtered_data[x_column_n]))
                                        fig.add_trace(go.Scatter(x=x_values, y=y_values, line_dash="dash",
                                                                name=f"Polynomial Trendline {degree} for {trace.name}",
                                                                line=dict(color=color))) 
                        
                        fig.update_layout(
                            title=f"Chart {i+1}",
                            xaxis_title=x_column_n,
                            yaxis_title="Value",
                            height=height,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)


                        
            
            # Append the chart summary data to the summary_data list
            chart_summary_data_10 = []

            for y_column in y_columns_n:
                data_row = {
                    "Group Data": 10,
                    "Chart": f"Chart {i+1}",
                    "Y Column": y_column,
                    "Min Value": filtered_data[y_column].min(),
                    "Max Value": filtered_data[y_column].max(),
                    "Average Value": filtered_data[y_column].mean(),
                    "Standard Deviation": filtered_data[y_column].std()
                }

                if trendline_n:
                    if trendline_type_n == "Linear":
                        data_row["Trendline Equation"] = f"y = {slope:.5f}x + {intercept:.5f}"
                        data_row["R-squared Value"] = f"{r_squared:.5f}"
                    elif trendline_type_n == "Polynomial":
                        data_row["Trendline Equation"] = f"y = {equation}"
                        data_row["R-squared Value"] = f"{r_squared:.5f}"

                chart_summary_data_10.append(data_row)

            summary_data_10.extend(chart_summary_data_10)  # Extend the summary data with the chart summary data

        # Convert the summary data into a DataFrame
        summary_df = pd.DataFrame(summary_data_10)
        st.session_state.summary_data_10 = summary_data_10

        # Display the summary DataFrame with automatic width adjustment
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.markdown("---")
        st.subheader("Summary Table")
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.warning("Please upload a CSV file first.")
