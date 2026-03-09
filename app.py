import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import jarque_bera

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="centered")
st.title("Time Series Forecasting Dashboard")

st.markdown("""
### Activity 1 – Time Series Forecasting

**Objective:**  
Fit multiple forecasting models, examine model adequacy using residual analysis,  
compare models, and forecast the next 5 time steps using the best model.
""")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("PJME_hourly.csv")

@st.cache_data
def load_residuals(file):
    return pd.read_csv(file).iloc[:, 0].dropna().iloc[-1000:]

@st.cache_data
def load_forecast(file):
    return pd.read_csv(file, index_col=0)

df = load_data()

# ---------------- DATASET OVERVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Time Series Plot (Last 1000 Observations)")
st.line_chart(df.iloc[-1000:, 1])

# ---------------- HELPER FUNCTION ----------------
def residual_diagnostics(residuals, model_name, conclusion):
    st.subheader(f"🔍 Residual Analysis – {model_name}")

    st.markdown("**1️⃣ Residuals vs Time**")
    st.line_chart(residuals)

    st.markdown("""
    Residuals fluctuate around zero with no systematic trend,
    indicating that the model captures the underlying structure.
    """)

    st.markdown("**2️⃣ ACF of Residuals**")
    fig, ax = plt.subplots(figsize=(6,3))
    plot_acf(residuals, lags=20, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    Most autocorrelations lie within the confidence bounds,
    suggesting that residuals behave like white noise.
    """)

    st.markdown("**3️⃣ Q–Q Plot of Residuals**")
    if st.checkbox(f"Show Q–Q Plot ({model_name})"):
        fig = sm.qqplot(residuals, line='s')
        st.pyplot(fig)

    st.markdown("**4️⃣ Normality Test (Jarque–Bera)**")
    jb_stat, jb_p = jarque_bera(residuals)
    st.write(f"Jarque–Bera p-value: {jb_p:.4f}")

    st.markdown("""
    A low p-value indicates deviation from normality,
    which is common in real-world energy demand data.
    """)

    st.markdown("**Model Adequacy Conclusion:**")
    st.write(conclusion)

# ---------------- SARIMA ----------------
sarima_resid = load_residuals("sarima_residuals.csv")

residual_diagnostics(
    sarima_resid,
    "SARIMA",
    """SARIMA residuals exhibit random fluctuations with low autocorrelation,
    indicating good model adequacy. However, the model involves higher
    parameter complexity and sensitivity to specification."""
)

# ---------------- HOLT–WINTERS ----------------
holt_resid = load_residuals("holt_residuals.csv")

residual_diagnostics(
    holt_resid,
    "Holt–Winters Exponential Smoothing",
    """Holt–Winters residuals show stable behavior with minimal structure,
    effectively capturing trend and strong weekly seasonality present
    in the energy consumption data."""
)

# ---------------- GARCH ----------------
garch_resid = load_residuals("garch_residuals.csv")

residual_diagnostics(
    garch_resid,
    "GARCH",
    """GARCH models time-varying volatility effectively; however,
    it is primarily suited for variance modeling rather than
    mean demand forecasting."""
)

# ---------------- MACHINE LEARNING ----------------
ml_resid = load_residuals("ml_residuals.csv")

residual_diagnostics(
    ml_resid,
    "Machine Learning",
    """The machine learning model captures nonlinear patterns but introduces
    additional complexity without improving short-term forecasting accuracy."""
)

# ---------------- RMSE COMPARISON ----------------
st.subheader("📐 RMSE Comparison")

rmse_df = pd.DataFrame({
    "Model": ["SARIMA", "Holt–Winters", "Machine Learning"],
    "RMSE": [1888.22, 1888.50, 2177.30]
})

st.table(rmse_df)

st.markdown("""
SARIMA and Holt–Winters exhibit nearly identical RMSE values,
indicating comparable forecasting accuracy.
""")

# ---------------- BEST MODEL ----------------
st.success("""
**Best Model Selected: Holt–Winters Exponential Smoothing**

**Justification:**
• Holt–Winters and SARIMA achieved nearly identical RMSE values  
• Holt–Winters directly models level, trend, and seasonality  
• Does not require strict stationarity assumptions  
• Simpler and more interpretable structure  
• Well-suited for short-term energy demand forecasting
""")

# ---------------- FORECAST ----------------
st.subheader("5-Step Ahead Forecast (Holt–Winters)")

forecast_df = load_forecast("Final_HoltWinters_5Step_Forecast.csv")
st.dataframe(forecast_df)

st.markdown("""
**Forecast Interpretation:**  
The Holt–Winters model produces stable and realistic short-term forecasts,
consistent with its strong residual diagnostics and seasonal structure.
""")

st.markdown("### Forecast Visualization")

if not forecast_df.empty:
    actual_series = df.iloc[-30:, 1]
    forecast_values = forecast_df.squeeze()

    forecast_index = range(len(actual_series),
                           len(actual_series) + len(forecast_values))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(len(actual_series)), actual_series,
            label="Actual (Last 30 Days)", marker='o')

    ax.plot(forecast_index, forecast_values,
            label="Holt–Winters Forecast (Next 5 Days)",
            marker='o', linestyle='--')

    ax.set_title("Holt–Winters: 5-Step Ahead Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Consumption")
    ax.legend()

    st.pyplot(fig)
else:
    st.warning("Forecast data is empty. Please check the forecast CSV file.")



st.subheader("Final Conclusion")

st.markdown("""
In this activity, a real-world hourly energy consumption dataset from the **PJME region**
was analyzed to forecast future energy demand. The data was preprocessed by converting
hourly observations into **daily averages** to ensure stability and interpretability of
the time series.

Multiple forecasting approaches were implemented, including **Holt–Winters Exponential
Smoothing**, **SARIMA**, **ARCH/GARCH**, and a **machine learning–based model**, in order
to capture different characteristics of the data.

Model adequacy was examined using **residual diagnostics**, including residual time plots,
ACF plots, Q–Q plots, and statistical tests where applicable. Holt–Winters and SARIMA were
primarily evaluated for **mean demand forecasting**, while ARCH/GARCH was used to model
**conditional volatility** and was therefore excluded from mean-based forecast comparison.
Machine learning models were also explored to assess potential non-linear patterns.

Model comparison was carried out using the **Root Mean Square Error (RMSE)** metric.
Holt–Winters and SARIMA produced **nearly identical RMSE values**, indicating comparable
forecasting accuracy. However, Holt–Winters was selected as the preferred model due to its
**simpler structure**, **direct handling of trend and seasonality**, and **stable residual
behavior**.

Using the selected Holt–Winters model, a **five-step (five-day) ahead forecast** of daily
energy consumption was generated. The forecasted values provide a reasonable estimation of
near-term energy demand trends and demonstrate the effectiveness of classical time series
techniques for short-term forecasting.

Overall, this study highlights the importance of **proper model selection**, **residual
diagnostics**, and **performance evaluation** in time series forecasting. Although
forecasting inherently involves uncertainty, the results indicate that **Holt–Winters
Exponential Smoothing** is well-suited for short-term energy consumption prediction in the
given dataset.
""")

