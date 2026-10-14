import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("historical_sales_raw.csv", parse_dates=["Date"])
df["Sales"] = df["Sales"].interpolate(limit_direction="both")
df["Week"] = np.arange(len(df))
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Lag_1"] = df["Sales"].shift(1)
df["Lag_4"] = df["Sales"].shift(4)
df["RollingMean_4"] = df["Sales"].rolling(4).mean()
df = df.dropna().reset_index(drop=True)
features = ["Week","Month","Quarter","Lag_1","Lag_4","RollingMean_4"]
split = int(len(df)*0.8)
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df["Sales"].iloc[:split], df["Sales"].iloc[split:]
model = RandomForestRegressor(n_estimators=200, random_state=21, max_depth=8)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test,pred),2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test,pred)),2))
print("R2:", round(r2_score(y_test,pred),3))
test = df.iloc[split:][["Date","Sales"]].copy()
test["PredictedSales"] = pred
test.to_csv("test_predictions.csv",index=False)
history = df["Sales"].tolist()
last = df["Date"].iloc[-1]
rows=[]
for step in range(1,13):
    dt=last+pd.Timedelta(weeks=step)
    row=pd.DataFrame([{"Week":len(df)+step-1,"Month":dt.month,"Quarter":((dt.month-1)//3)+1,"Lag_1":history[-1],"Lag_4":history[-4],"RollingMean_4":np.mean(history[-4:])}])
    value=float(model.predict(row[features])[0])
    rows.append({"Date":dt,"ForecastSales":value})
    history.append(value)
forecast=pd.DataFrame(rows)
forecast.to_csv("next_12_weeks_forecast.csv",index=False)
plt.plot(df["Date"],df["Sales"],label="Historical")
plt.plot(test["Date"],pred,label="Test predictions")
plt.title("Historical Sales vs Predictions"); plt.legend(); plt.show()
plt.plot(forecast["Date"],forecast["ForecastSales"],marker="o",label="12-week forecast")
plt.title("Next 12 Weeks Forecast"); plt.legend(); plt.show()
