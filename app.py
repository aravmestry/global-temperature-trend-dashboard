import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Global Temperature Trend Dashboard",
    page_icon="🌡️",
    layout="wide"
)

DATA_PATH = "data/02_berkeley_earth_global_temperature.txt"

@st.cache_data
def load_data():
    rows = []

    with open(DATA_PATH, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    anomaly = float(parts[2])
                    uncertainty = float(parts[3])

                    if 1800 <= year <= 2100 and 1 <= month <= 12:
                        rows.append([year, month, anomaly, uncertainty])
                except:
                    pass

    df = pd.DataFrame(
        rows,
        columns=["year", "month", "anomaly_C", "uncertainty_C"]
    )

    df["decimal_year"] = df["year"] + (df["month"] - 0.5) / 12

    annual = df.groupby("year", as_index=False).agg(
        anomaly_C=("anomaly_C", "mean"),
        uncertainty_C=("uncertainty_C", "mean")
    )

    annual["rolling_10yr"] = annual["anomaly_C"].rolling(10).mean()

    return df, annual

df, annual = load_data()

X = annual[["year"]]
y = annual["anomaly_C"]

linear = LinearRegression()
linear.fit(X, y)
annual["linear_trend"] = linear.predict(X)

quad = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
quad.fit(X, y)
annual["quadratic_trend"] = quad.predict(X)

recent = annual[annual["year"] >= 1970]
recent_model = LinearRegression()
recent_model.fit(recent[["year"]], recent["anomaly_C"])
warming_rate_decade = recent_model.coef_[0] * 10

future_years = pd.DataFrame({"year": np.arange(annual["year"].max() + 1, 2101)})
future_years["projected_anomaly_C"] = quad.predict(future_years[["year"]])

st.title("🌡️ Global Temperature Trend Dashboard")

st.markdown(
    """
    This dashboard analyzes global surface temperature anomalies using the Berkeley Earth
    global land/ocean temperature record. It compares linear and nonlinear warming trends,
    estimates the recent warming rate, and projects future anomaly trajectories to 2100.
    """
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Years Covered", f"{annual['year'].min()}–{annual['year'].max()}")
m2.metric("Annual Records", f"{len(annual)}")
m3.metric("Warming Since 1970", f"{warming_rate_decade:.3f} °C/decade")
m4.metric("Latest Anomaly", f"{annual['anomaly_C'].iloc[-1]:.2f} °C")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend Analysis",
    "🔮 Future Projection",
    "📊 Period Comparison",
    "📁 Data Explorer"
])

with tab1:
    st.subheader("Global Temperature Anomaly Trend")

    plot_df = annual.melt(
        id_vars="year",
        value_vars=["anomaly_C", "rolling_10yr", "linear_trend", "quadratic_trend"],
        var_name="Series",
        value_name="Temperature Anomaly (°C)"
    )

    fig = px.line(
        plot_df,
        x="year",
        y="Temperature Anomaly (°C)",
        color="Series",
        title="Global Temperature Anomalies and Modeled Trends"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Projected Temperature Anomaly to 2100")

    fig_future = px.line(
        future_years,
        x="year",
        y="projected_anomaly_C",
        title="Quadratic Projection of Global Temperature Anomaly to 2100",
        labels={
            "year": "Year",
            "projected_anomaly_C": "Projected Temperature Anomaly (°C)"
        }
    )

    st.plotly_chart(fig_future, use_container_width=True)

    st.warning(
        "This projection is a simple statistical extrapolation, not a full climate model. "
        "It should be interpreted as a trend-based scenario, not a physical climate forecast."
    )

with tab3:
    st.subheader("Warming Rate by Period")

    periods = {
        "1850–1900": annual[(annual["year"] >= 1850) & (annual["year"] <= 1900)],
        "1901–1950": annual[(annual["year"] >= 1901) & (annual["year"] <= 1950)],
        "1951–1970": annual[(annual["year"] >= 1951) & (annual["year"] <= 1970)],
        "1971–Present": annual[annual["year"] >= 1971]
    }

    period_rows = []

    for name, subset in periods.items():
        if len(subset) > 5:
            model = LinearRegression()
            model.fit(subset[["year"]], subset["anomaly_C"])
            period_rows.append({
                "Period": name,
                "Warming Rate (°C/decade)": model.coef_[0] * 10,
                "Mean Anomaly (°C)": subset["anomaly_C"].mean()
            })

    period_df = pd.DataFrame(period_rows)

    st.dataframe(period_df, use_container_width=True)

    fig_period = px.bar(
        period_df,
        x="Period",
        y="Warming Rate (°C/decade)",
        title="Estimated Warming Rate by Historical Period"
    )

    st.plotly_chart(fig_period, use_container_width=True)

with tab4:
    st.subheader("Annual Temperature Dataset")
    st.dataframe(annual, use_container_width=True)

    csv = annual.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Annual Temperature Data",
        data=csv,
        file_name="annual_temperature_anomalies.csv",
        mime="text/csv"
    )