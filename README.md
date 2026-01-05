# ⚡ Time Series Prediction of Monthly Electricity Consumption using RNN (LSTM)

I implemented a **time-series forecasting model** to predict **monthly electricity consumption** using a **Recurrent Neural Network (LSTM)**. Since electricity usage depends on historical patterns, sequence-based deep learning models are more suitable than traditional machine learning approaches.

---

## 🎯 Objective
- Analyze historical monthly electricity consumption
- Capture trends and seasonal patterns
- Predict future electricity usage

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  

---

## 📂 Dataset
The dataset (`electricity_consumption.csv`) contains:
- **Bill_Date** – monthly timestamps  
- **Billed_amount** – electricity consumption values  

This is a **univariate time-series dataset**.

---

## 🔄 Methodology

### 1️⃣ Data Visualization
I visualized the electricity consumption data to understand overall trends and fluctuations.

<img width="852" height="520" alt="Screenshot 2026-01-05 194029" src="https://github.com/user-attachments/assets/5d371674-c18e-476d-9987-b33f7eb93c02" />


### **2️⃣ Time Series Decomposition**

To analyze trend, seasonality, and residual components, I applied seasonal decomposition.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(df['Billed_amount'])
results.plot()
```
This helped confirm that the data contains trend and seasonal components, making it suitable for sequence modeling.

### **3️⃣ Data Scaling**

I normalized the data using MinMaxScaler to improve neural network training.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Billed_amount']])
```
<img width="874" height="600" alt="Screenshot 2026-01-05 194017" src="https://github.com/user-attachments/assets/9c092db1-a9eb-4d25-ab47-a15ed81663cd" />

### **4️⃣ Sequence Generation (Sliding Window)**

I converted the time series into supervised learning format using ```TimeseriesGenerator.```
```python
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

n_input = 3
generator = TimeseriesGenerator(
    scaled_data, scaled_data,
    length=n_input,
    batch_size=1
)
```
Each input sequence predicts the next month’s consumption.

### **5️⃣ Model Architecture (LSTM)**

I used an LSTM-based RNN to capture temporal dependencies.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(100, activation='relu', input_shape=(n_input, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```
LSTM was chosen over a simple RNN due to its ability to handle longer-term dependencies.

<img width="863" height="688" alt="Screenshot 2026-01-05 193954" src="https://github.com/user-attachments/assets/492df023-6756-4582-bafe-9d4c054d2abe" />

### **6️⃣ Model Training**

The model was trained using the generated sequences.
```python
model.fit(generator, epochs=50)
```
<img width="1290" height="686" alt="Screenshot 2026-01-05 193917" src="https://github.com/user-attachments/assets/7d655a36-7765-462b-a8c3-3bbdcbb70ffe" />

### **7️⃣ Prediction & Evaluation**

I generated predictions on test data and compared them with actual values.
```python
test.plot(figsize=(14,5))
```
To evaluate performance numerically, I calculated RMSE.

```python
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(
    test['Billed_amount'],
    test['Predictions']
))
print(rmse)
```
<img width="1288" height="718" alt="Screenshot 2026-01-05 193844" src="https://github.com/user-attachments/assets/24a93e8e-8136-4691-9775-5606d0bae5e8" />

---

## **📊 Results**

-The model successfully learned consumption patterns

-Predictions closely follow actual electricity usage

-RMSE indicates reasonable forecasting accuracy

So, the next billed amount is found to be ₹ 65.18
