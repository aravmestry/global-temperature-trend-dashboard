import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "data/02_berkeley_earth_global_temperature.txt"

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

df = pd.DataFrame(rows, columns=["year", "month", "anomaly_C", "uncertainty_C"])
df["decimal_year"] = df["year"] + (df["month"] - 0.5) / 12

annual = df.groupby("year", as_index=False).agg(
    anomaly_C=("anomaly_C", "mean"),
    uncertainty_C=("uncertainty_C", "mean")
)

X = annual[["year"]]
y = annual["anomaly_C"]

linear = LinearRegression()
linear.fit(X, y)
linear_pred = linear.predict(X)

poly = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly.fit(X, y)
poly_pred = poly.predict(X)

def metrics(name, pred):
    return {
        "Model": name,
        "R2": round(r2_score(y, pred), 4),
        "MAE": round(mean_absolute_error(y, pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 4)
    }

results = pd.DataFrame([
    metrics("Linear Regression", linear_pred),
    metrics("Quadratic Regression", poly_pred)
])

results.to_csv("outputs/model_comparison.csv", index=False)
annual.to_csv("outputs/annual_temperature_anomalies.csv", index=False)

print("\nMODEL COMPARISON")
print(results)

recent = annual[annual["year"] >= 1970]
X_recent = recent[["year"]]
y_recent = recent["anomaly_C"]

recent_model = LinearRegression()
recent_model.fit(X_recent, y_recent)

warming_rate_per_year = recent_model.coef_[0]
warming_rate_per_decade = warming_rate_per_year * 10

print("\nWARMING RATE SINCE 1970")
print(f"{warming_rate_per_decade:.3f} °C per decade")

plt.figure(figsize=(10, 5))
plt.plot(annual["year"], annual["anomaly_C"], label="Annual anomaly")
plt.plot(annual["year"], linear_pred, label="Linear trend")
plt.plot(annual["year"], poly_pred, label="Quadratic trend")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Global Temperature Anomaly Trend")
plt.legend()
plt.tight_layout()
plt.savefig("figures/01_temperature_trend.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["decimal_year"], df["anomaly_C"], alpha=0.35)
plt.xlabel("Year")
plt.ylabel("Monthly Temperature Anomaly (°C)")
plt.title("Monthly Global Temperature Anomalies")
plt.tight_layout()
plt.savefig("figures/02_monthly_temperature_anomalies.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["R2"])
plt.ylabel("R²")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("figures/03_model_comparison.png", dpi=300)
plt.close()

print("\nSaved outputs and figures.")