import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Defense Maintenance - RUL Prediction",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Defense Maintenance Prediction System")
st.subheader("Remaining Useful Life (RUL) Prediction")

st.markdown("""
Bu uygulama, motor sensör verilerini kullanarak
**Remaining Useful Life (RUL)** tahmini yapar.
""")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "22.62")

with col2:
    st.metric("RMSE", "29.59")

with col3:
    st.metric("R² Score", "0.797")

st.info(
    "Gradient Boosting modeli en iyi performansı göstermiştir."
)

# -----------------------------
# MODEL COMPARISON
# -----------------------------

st.header("🤖 Model Comparison")

model_data = pd.DataFrame({
    "Model": [
        "Gradient Boosting",
        "Random Forest",
        "Sensor Only"
    ],
    "MAE": [
        22.624,
        23.893,
        29.887
    ],
    "RMSE": [
        29.592,
        31.499,
        41.335
    ],
    "R²": [
        0.797,
        0.770,
        0.626
    ]
})

st.dataframe(
    model_data,
    use_container_width=True
)

# -----------------------------
# ACTUAL VS PREDICTED
# -----------------------------

st.header("📈 Actual vs Predicted RUL")

try:
    st.image(
        "actual_vs_predicted.png",
        caption="Actual vs Predicted RUL",
        use_container_width=True
    )
except:
    st.warning("Actual vs Predicted grafiği bulunamadı.")

# -----------------------------
# SENSOR IMPORTANCE
# -----------------------------

st.header("🔍 Feature Importance")

feature_data = pd.DataFrame({
    "Feature": [
        "cycle",
        "sensor_11",
        "sensor_4",
        "sensor_12",
        "sensor_7",
        "sensor_9",
        "sensor_15",
        "sensor_14",
        "sensor_21",
        "sensor_2",
        "sensor_20",
        "sensor_3",
        "sensor_8",
        "sensor_17",
        "sensor_13"
    ],
    "Importance": [
        0.683,
        0.089,
        0.069,
        0.029,
        0.026,
        0.026,
        0.018,
        0.017,
        0.014,
        0.010,
        0.006,
        0.004,
        0.004,
        0.003,
        0.002
    ]
})

st.bar_chart(
    feature_data.set_index("Feature")
)

# -----------------------------
# PREDICTION ERROR
# -----------------------------

st.header("📉 Prediction Error Analysis")

error_col1, error_col2, error_col3 = st.columns(3)

with error_col1:
    st.metric("Mean Error", "-3.11")

with error_col2:
    st.metric("Mean Absolute Error", "22.62")

with error_col3:
    st.metric("Maximum Error", "84.29")

st.success("Prediction analysis completed successfully.")

# -----------------------------
# GROUP VALIDATION
# -----------------------------

st.header("🧪 Group K-Fold Cross Validation")

cv_data = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "MAE": [
        28.287,
        25.902,
        29.068,
        27.058,
        28.269
    ]
})

st.dataframe(cv_data, use_container_width=True)

st.metric(
    "Average Group MAE",
    "27.72"
)

st.metric(
    "Standard Deviation",
    "1.11"
)

st.divider()

st.caption(
    "Defense Maintenance Prediction Project | RUL Prediction"
)




import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Defense Maintenance Prediction",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Defense Maintenance Prediction System")
st.subheader("Remaining Useful Life (RUL) Prediction")

st.write(
    "Bu uygulama, motor sensör verilerini kullanarak "
    "Remaining Useful Life (RUL) tahmini yapar."
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("processed/engine_maintenance.csv")


data = load_data()


# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------

feature_columns = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
]

sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

feature_columns += sensor_columns

X = data[feature_columns]
y = data["RUL"]


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return model, mae, rmse, r2


model, mae, rmse, r2 = train_model(X, y)


# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

with col3:
    st.metric("R² Score", f"{r2:.3f}")


st.info(
    "Gradient Boosting modeli kullanılarak motorun "
    "kalan faydalı ömrü (RUL) tahmin edilmektedir."
)


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

st.header("🔮 RUL Prediction")

st.write(
    "Motor bilgilerini girerek kalan faydalı ömrü tahmin edebilirsiniz."
)


# Cycle
cycle = st.number_input(
    "Cycle",
    min_value=1,
    max_value=int(data["cycle"].max()),
    value=100
)


# Settings
st.subheader("Motor Settings")

setting_1 = st.number_input(
    "Setting 1",
    value=float(data["setting_1"].mean())
)

setting_2 = st.number_input(
    "Setting 2",
    value=float(data["setting_2"].mean())
)

setting_3 = st.number_input(
    "Setting 3",
    value=float(data["setting_3"].mean())
)


# Sensors
st.subheader("Sensor Values")

sensor_values = {}

cols = st.columns(3)

for i, sensor in enumerate(sensor_columns):

    with cols[i % 3]:

        sensor_values[sensor] = st.number_input(
            sensor,
            value=float(data[sensor].mean())
        )


# -------------------------------------------------
# MAKE PREDICTION
# -------------------------------------------------

if st.button("🚀 Predict RUL"):

    input_data = {
        "cycle": cycle,
        "setting_1": setting_1,
        "setting_2": setting_2,
        "setting_3": setting_3
    }

    input_data.update(sensor_values)

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    prediction = max(0, prediction)

    st.success(
        f"Predicted Remaining Useful Life: **{prediction:.2f} cycles**"
    )

    if prediction < 20:
        st.error("⚠️ Maintenance is recommended soon.")

    elif prediction < 50:
        st.warning("🟡 Maintenance should be planned.")

    else:
        st.success("🟢 Engine condition appears relatively healthy.")


# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.header("📈 Feature Importance")

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)