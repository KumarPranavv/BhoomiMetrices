import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

# -----------------------------------------------------------------------------
# Set page config (title, icon, layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BhoomiMetrices: Land Price Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"  # 'wide' layout for side-by-side charts
)

# -----------------------------------------------------------------------------
# Center the logo with custom HTML
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("Images/LandLogo.png", width=400)

# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>BhoomiMetrices - Land Price Prediction (1-Year & 2-Year)</h1>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Create Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Forecast", "Data Preview", "Model Info"])

# -----------------------------------------------------------------------------
# TAB 1: FORECAST / PREDICTION
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Enter Current Details for Forecast")

    # 1) Load Models
    model_1yr = joblib.load("models/model_1yr.pkl")
    model_2yr = joblib.load("models/model_2yr.pkl")

    # 2) Collect User Inputs
    city_choice = st.selectbox(
        "Select City:",
        ["Gurugram", "Pune", "Nagpur", "Ujjain", "Meerut", "Haridwar", "Noida"],
        key="city_choice_key"
    )

    price_input = st.number_input(
        "Current Land Price(in INR) (per sq ft):",
        value=5000.0,
        step=500.0,
        key="price_input_key"
    )

    base_year_input = st.number_input(
        "Base Year (YYYY) for Forecast:",
        value=2025,
        min_value=2000,
        max_value=2100,
        key="base_year_key"
    )

    base_month_input = st.number_input(
        "Base Month (1-12) for Forecast:",
        value=1,
        min_value=1,
        max_value=12,
        key="base_month_key"
    )

    st.write("Now, just click on 'Predict' ")

    # 3) Predict Button
    if st.button("Predict", key="predict_main"):
        # ---------------------------------------------------------------------
        # SINGLE-POINT PREDICTION
        # ---------------------------------------------------------------------
        df_input = pd.DataFrame([{
            "price": price_input,
            "year": base_year_input,
            "month": base_month_input,
            "city": city_choice
        }])

        # Convert city to dummies & align columns for 1yr model
        city_dummies = pd.get_dummies(df_input["city"], prefix="city")
        X_input = pd.concat([df_input[["price", "year", "month"]], city_dummies], axis=1)

        required_cols_1yr = model_1yr.feature_names_in_
        for col in required_cols_1yr:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input_1yr = X_input[required_cols_1yr]

        # Convert city & align columns for 2yr model (could be same or slight differences)
        required_cols_2yr = model_2yr.feature_names_in_
        for col in required_cols_2yr:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input_2yr = X_input[required_cols_2yr]

        # Predict single point
        pred_1yr = model_1yr.predict(X_input_1yr)[0]
        pred_2yr = model_2yr.predict(X_input_2yr)[0]

        st.subheader("Single-Point Forecast")
        # Show predictions side-by-side with st.metric
        single_col1, single_col2 = st.columns(2)
        with single_col1:
            st.metric("1-Year Price", f"{pred_1yr:,.2f} per sq ft")
        with single_col2:
            st.metric("2-Year Price", f"{pred_2yr:,.2f} per sq ft")

        st.success("Prediction complete!")

        # ---------------------------------------------------------------------
        # 12-MONTH FORECAST (1-Year Model)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("12-Month Forecast (1-Year Model)")

        future_range_1yr = 12
        start_date_1yr = datetime.date(base_year_input, base_month_input, 1)

        future_data_1yr = []
        for m in range(future_range_1yr):
            future_month = start_date_1yr + relativedelta(months=m)
            future_data_1yr.append({
                "year": future_month.year,
                "month": future_month.month,
                "price": price_input,
                "city": city_choice
            })
        df_future_1yr = pd.DataFrame(future_data_1yr)

        # Convert city & align columns
        city_dummies_future_1yr = pd.get_dummies(df_future_1yr["city"], prefix="city")
        X_future_1yr = pd.concat([df_future_1yr[["price", "year", "month"]], city_dummies_future_1yr], axis=1)

        for col in required_cols_1yr:
            if col not in X_future_1yr.columns:
                X_future_1yr[col] = 0
        X_future_1yr = X_future_1yr[required_cols_1yr]

        future_preds_1yr = model_1yr.predict(X_future_1yr)
        df_future_1yr["predicted_price"] = future_preds_1yr

        # Create a datetime index for plotting
        date_list_1yr = [
            datetime.date(row["year"], row["month"], 1)
            for _, row in df_future_1yr.iterrows()
        ]
        df_future_1yr["date"] = date_list_1yr
        df_future_1yr.set_index("date", inplace=True)

        # Format the predicted_price
        df_1yr_styled = df_future_1yr[["year", "month", "predicted_price"]].style.format(
            {"predicted_price": "{:.2f}"}
        )

        st.dataframe(df_1yr_styled, use_container_width=True)

        # Show line chart & bar chart side by side
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Line Chart (1-Year Forecast)**")
            st.line_chart(df_future_1yr["predicted_price"])
        with c2:
            st.markdown("**Bar Chart (1-Year Forecast)**")
            st.bar_chart(df_future_1yr["predicted_price"])

        # ---------------------------------------------------------------------
        # 24-MONTH FORECAST (2-Year Model)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("24-Month Forecast (2-Year Model)")

        future_range_2yr = 24
        start_date_2yr = datetime.date(base_year_input, base_month_input, 1)

        future_data_2yr = []
        for m in range(future_range_2yr):
            future_month = start_date_2yr + relativedelta(months=m)
            future_data_2yr.append({
                "year": future_month.year,
                "month": future_month.month,
                "price": price_input,
                "city": city_choice
            })
        df_future_2yr = pd.DataFrame(future_data_2yr)

        # Convert city & align columns
        city_dummies_future_2yr = pd.get_dummies(df_future_2yr["city"], prefix="city")
        X_future_2yr = pd.concat([df_future_2yr[["price", "year", "month"]], city_dummies_future_2yr], axis=1)

        for col in required_cols_2yr:
            if col not in X_future_2yr.columns:
                X_future_2yr[col] = 0
        X_future_2yr = X_future_2yr[required_cols_2yr]

        future_preds_2yr = model_2yr.predict(X_future_2yr)
        df_future_2yr["predicted_price"] = future_preds_2yr

        date_list_2yr = [
            datetime.date(row["year"], row["month"], 1)
            for _, row in df_future_2yr.iterrows()
        ]
        df_future_2yr["date"] = date_list_2yr
        df_future_2yr.set_index("date", inplace=True)

        df_2yr_styled = df_future_2yr[["year", "month", "predicted_price"]].style.format(
            {"predicted_price": "{:.2f}"}
        )

        st.dataframe(df_2yr_styled, use_container_width=True)

        # Show line chart & bar chart side by side
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Line Chart (2-Year Forecast)**")
            st.line_chart(df_future_2yr["predicted_price"])
        with c4:
            st.markdown("**Bar Chart (2-Year Forecast)**")
            st.bar_chart(df_future_2yr["predicted_price"])
    st.markdown("---")
    st.write("© 2025 | Pranav Kumar. All Rights Reserved.")


# -----------------------------------------------------------------------------
# TAB 2: DATA PREVIEW
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Data Preview / Sample Data")
    st.write(
        "This is the snippet of the Processed data! "
        
    )

    if st.button("Show Sample Data", key="show_sample_data"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("Images/Processed_data_snippet.png", width=650)

    st.markdown("---")
    st.write("© 2025 | Pranav Kumar. All Rights Reserved.")

# -----------------------------------------------------------------------------
# TAB 3: MODEL INFO
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model & Project Information")

    st.write("**Project Name**: BhoomiMetrices")
    st.write("**Objective**: Provide 1-year and 2-year land price forecasts based on current land price, location, Inflation Rate, CDI and date.")

    st.write("### Brief Description of the Model")
    st.write(
        "We use two separate **Random Forest** regressors, each trained on historical land price data:\n"
        "- **1-Year Model**: Predicts land price 12 months ahead.\n"
        "- **2-Year Model**: Predicts land price 24 months ahead.\n\n"
        "These models incorporate features such as **current price**, **month**, **year**, and **city** (one-hot encoded). "
        "Random Forest was chosen for its robustness to outliers and ability to handle complex interactions. "
    )

    st.write("### Future Enhancements")
    st.write(
        "- Expand city list to cover more regions.\n"
        "- Integrate advanced data for **5-year** or longer-range predictions.\n"
        "- Add macroeconomic indicators such as inflation, interest rates, and GDP growth.\n"
        "- Provide user-friendly CSV upload for batch predictions."
    )

    st.markdown("---")
    st.write("**Connect with Me**:")
    st.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/kumar-pranavv/)", unsafe_allow_html=True)
    st.markdown("[Follow me on GitHub](https://github.com/KumarPranavv)", unsafe_allow_html=True)

   

    st.info("For further suggestions or collaboration or questions, feel free to reach out at kumar2pranav@gmail.com")
    
    st.markdown("---")
    st.write("© 2025 | Pranav Kumar. All Rights Reserved.")

