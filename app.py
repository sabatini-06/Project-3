
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    # Example data. Replace this with actual data loading if needed
    data = {
        'Area': ['China'] * 5,
        'Item': ['Apricots'] * 5,
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Production_tonnes': [70000, 72000, 74000, 76000, 78000]
    }
    return pd.DataFrame(data)

df = load_data()

st.title("🌾 Crop Production Forecast Dashboard")

crop_list = df['Item'].unique()
area_list = df['Area'].unique()

selected_crop = st.selectbox("Select Crop", sorted(crop_list))
selected_area = st.selectbox("Select Area", sorted(area_list))

model_data = df[(df['Item'] == selected_crop) & (df['Area'] == selected_area)].copy()

if model_data.shape[0] < 2:
    st.warning(f"Not enough data points for {selected_crop} in {selected_area} to build a model.")
else:
    X = model_data[['Year']].values
    y = model_data['Production_tonnes'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred) if np.var(y) != 0 else float('nan')

    last_year = model_data['Year'].max()
    future_years = np.array([[year] for year in range(last_year + 1, last_year + 6)])
    future_preds = model.predict(future_years)

    st.subheader("📊 Model Evaluation")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {'NaN' if np.isnan(r2) else f'{r2:.2f}'}")

    st.subheader("📈 Future Production Forecast")
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Production (tonnes)': [max(0, p) for p in future_preds]
    })
    st.dataframe(forecast_df)

    st.subheader("📉 Production Trend")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y, label='Historical Data', color='green')
    ax.plot(X, y_pred, color='blue', label='Linear Fit')
    ax.plot(future_years, future_preds, color='orange', linestyle='--', label='Future Prediction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Production (tonnes)')
    ax.set_title(f'Production Forecast: {selected_crop} in {selected_area}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
