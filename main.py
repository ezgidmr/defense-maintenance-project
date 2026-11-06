from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import sqlite3

# 1. Veri dosyasını bul
files = list(Path.cwd().rglob("train_FD001.txt"))

if not files:
    raise FileNotFoundError("train_FD001.txt bulunamadı!")

file_path = files[0]

# 2. Sütun isimlerini oluştur
columns = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# 3. Veri dosyasını oku
data = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# 4. İlk 5 satırı göster
print("\nİLK 5 SATIR:")
print(data.head())

# 5. Veri boyutunu göster
print("\nVERİ BOYUTU:")
print(data.shape)

# 6. Sütun isimlerini göster
print("\nSÜTUN İSİMLERİ:")
print(data.columns.tolist())

print("\n--- VERİ ANALİZİ ---")

# Toplam satır ve sütun sayısı
print("Veri boyutu:", data.shape)

# Kaç farklı motor var?
print("Motor sayısı:", data["unit_id"].nunique())

# Eksik değer sayısı
print("Eksik değer sayısı:", data.isnull().sum().sum())

# Birinci motorun verilerini seç
motor_1 = data[data["unit_id"] == 1]

print("\n--- MOTOR 1 ANALİZİ ---")

# Motorun kaç döngü çalıştığını göster
print("Çalışma döngüsü:", motor_1["cycle"].max())

# İlk 5 döngüyü göster
print("\nİlk 5 ölçüm:")
print(motor_1.head())

# Son 5 döngüyü göster
print("\nSon 5 ölçüm:")
print(motor_1.tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    motor_1["cycle"],
    motor_1["sensor_2"]
)

plt.title("Engine 1 - Sensor 2 Over Time")
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor 2 Value")

plt.grid(True)

print("\n--- ALL SENSOR STATISTICS ---")

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

sensor_statistics = data[sensor_columns].describe().T

print(sensor_statistics)

plt.savefig(
    "engine_lifetime_distribution.png",
    dpi=300,

    bbox_inches="tight"
)



plt.show()



print("\n--- SENSOR 2 STATISTICS ---")

print("Average:", motor_1["sensor_2"].mean())

print("Minimum:", motor_1["sensor_2"].min())

print("Maximum:", motor_1["sensor_2"].max())

print("Standard Deviation:", motor_1["sensor_2"].std())

# Bütün sensörlerin istatistikleri

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

print("\n--- ALL SENSOR STATISTICS ---")

sensor_statistics = data[sensor_columns].describe().T

print(sensor_statistics)


# Sensörleri standart sapmaya göre sırala

sensor_ranking = sensor_statistics.sort_values(
    by="std",
    ascending=False
)

print("\n--- SENSOR VARIABILITY RANKING ---")

print(sensor_ranking[["mean", "std"]])


# Motorların son çalışma döngülerini bul

last_cycles = data.groupby("unit_id")["cycle"].max()

# Her motor için RUL hesapla

data["RUL"] = (
    data["unit_id"].map(last_cycles) - data["cycle"]
)

print("\n--- RUL ANALYSIS ---")

# İlk motorun ilk 5 RUL değerini göster

print(
    data[data["unit_id"] == 1][
        ["unit_id", "cycle", "RUL"]
    ].head()
)

# İşlenmiş veriyi CSV olarak kaydet

output_folder = Path("processed")
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "engine_maintenance.csv"

data.to_csv(output_file, index=False)

print("\n--- DATA EXPORT ---")
print("Veri başarıyla kaydedildi!")
print("Dosya:", output_file)
# SQL DATABASE

connection = sqlite3.connect("maintenance.db")

data.to_sql(
    "engine_data",
    connection,
    if_exists="replace",
    index=False
)

print("\n--- SQL DATABASE ---")
print("Database created successfully!")

connection.close()

# SQL DATA CHECK

connection = sqlite3.connect("maintenance.db")

cursor = connection.cursor()

cursor.execute("""
SELECT COUNT(*)
FROM engine_data
""")

result = cursor.fetchone()

print("\n--- SQL DATA CHECK ---")
print("Total records:", result[0])

connection.close()

# MOTOR BAZINDA SQL ANALİZİ

connection = sqlite3.connect("maintenance.db")

query = """
SELECT
    unit_id,
    COUNT(*) AS total_records,
    MAX(cycle) AS total_cycles,
    AVG(RUL) AS average_rul
FROM engine_data
GROUP BY unit_id
ORDER BY unit_id
"""

motor_analysis = pd.read_sql_query(query, connection)

print("\n--- MOTOR ANALYSIS ---")
print(motor_analysis.head(10))

connection.close()

# EN UZUN VE EN KISA CALISAN MOTORLAR

connection = sqlite3.connect("maintenance.db")

query = """
SELECT
    unit_id,
    MAX(cycle) AS total_cycles
FROM engine_data
GROUP BY unit_id
ORDER BY total_cycles DESC
"""

result = pd.read_sql_query(query, connection)

print("\n--- ENGINE LIFETIME ANALYSIS ---")

print("En uzun çalışan 5 motor:")
print(result.head(5))

print("\nEn kısa çalışan 5 motor:")
print(result.tail(5))

connection.close()

# MOTOR CALISMA SURELERI GRAFIGI

plt.figure(figsize=(10, 5))

plt.hist(
    result["total_cycles"],
    bins=15,
    edgecolor="black"
)

plt.title("Engine Lifetime Distribution")
plt.xlabel("Total Operating Cycles")
plt.ylabel("Number of Engines")

plt.grid(axis="y", alpha=0.3)

plt.show()


# MACHINE LEARNING - DATA PREPARATION

# Modelin tahmin edeceği hedef
y = data["RUL"]

# Modelin öğreneceği özellikler
X = data.drop(columns=["unit_id", "RUL"])

print("\n--- MACHINE LEARNING DATA ---")

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())


# TRAIN TEST SPLIT

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=data["unit_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\n--- TRAIN TEST SPLIT ---")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("Training engines:", data.iloc[train_idx]["unit_id"].nunique())
print("Testing engines:", data.iloc[test_idx]["unit_id"].nunique())


# RANDOM FOREST MODEL

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("\n--- MODEL TRAINING ---")

model.fit(X_train, y_train)

print("Model training completed!")


# MODEL EVALUATION

print("\n--- MODEL EVALUATION ---")

# Test verileriyle tahmin yap
y_pred = model.predict(X_test)

# Performans metrikleri
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

print("\nModel evaluation completed!")


# REAL VS PREDICTED RUL

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title("Actual vs Predicted RUL")

plt.grid(True)

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n--- RANDOM FOREST MODEL ---")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2:", rf_r2)



# GRADIENT BOOSTING MODEL

from sklearn.ensemble import GradientBoostingRegressor

print("\n--- GRADIENT BOOSTING MODEL ---")

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)


import joblib

joblib.dump(gb_model, "gradient_boosting_model.pkl")
print("Model saved successfully!")

gb_mae = mean_absolute_error(y_test, gb_pred)

gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

gb_r2 = r2_score(y_test, gb_pred)

print("Gradient Boosting MAE:", gb_mae)
print("Gradient Boosting RMSE:", gb_rmse)
print("Gradient Boosting R2:", gb_r2)




# SENSOR IMPORTANCE ANALYSIS

print("\n--- SENSOR IMPORTANCE ---")

importance = pd.DataFrame({
    "Sensor": X.columns,
    "Importance": gb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

# Grafik

plt.figure(figsize=(10, 7))

plt.barh(
    importance.head(15)["Sensor"],
    importance.head(15)["Importance"]
)

plt.xlabel("Importance")
plt.title("Top 15 Important Sensors")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("sensor_importance.png")

plt.show()


# SENSOR-ONLY MODEL

print("\n--- SENSOR ONLY MODEL ---")

# Cycle değişkenini çıkarıyoruz
X_sensor = X.drop(columns=["cycle"])

# Eğitim ve test verilerini aynı şekilde ayır
from sklearn.model_selection import train_test_split

X_train_sensor, X_test_sensor, y_train_sensor, y_test_sensor = train_test_split(
    X_sensor,
    y,
    test_size=0.2,
    random_state=42
)

# Yeni model
sensor_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

sensor_model.fit(X_train_sensor, y_train_sensor)

sensor_pred = sensor_model.predict(X_test_sensor)

print("Sensor Only MAE:",
      mean_absolute_error(y_test_sensor, sensor_pred))

print("Sensor Only RMSE:",
      np.sqrt(mean_squared_error(y_test_sensor, sensor_pred)))

print("Sensor Only R2:",
      r2_score(y_test_sensor, sensor_pred))


# GROUP BASED VALIDATION
# Motorları birbirinden ayırarak test ediyoruz.

from sklearn.model_selection import GroupShuffleSplit

print("\n--- GROUP BASED VALIDATION ---")

groups = data["unit_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train_group = X.iloc[train_idx]
X_test_group = X.iloc[test_idx]

y_train_group = y.iloc[train_idx]
y_test_group = y.iloc[test_idx]

# Model eğitimi

group_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

group_model.fit(X_train_group, y_train_group)

group_pred = group_model.predict(X_test_group)

print("Group MAE:",
      mean_absolute_error(y_test_group, group_pred))

print("Group RMSE:",
      np.sqrt(mean_squared_error(y_test_group, group_pred)))

print("Group R2:",
      r2_score(y_test_group, group_pred))

print("Training engines:",
      groups.iloc[train_idx].nunique())

print("Testing engines:",
      groups.iloc[test_idx].nunique())


# VALIDATION CHECK

print("\n--- VALIDATION CHECK ---")

print("Train index first 10:",
      train_idx[:10])

print("Test index first 10:",
      test_idx[:10])

print("Train engines:",
      sorted(groups.iloc[train_idx].unique()))

print("Test engines:",
      sorted(groups.iloc[test_idx].unique()))

print("Same engine in both sets:",
      set(groups.iloc[train_idx].unique()).intersection(
          set(groups.iloc[test_idx].unique())
      ))


# MODEL OPTIMIZATION

from sklearn.model_selection import GridSearchCV

print("\n--- MODEL OPTIMIZATION ---")

param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best MAE:", -grid_search.best_score_)

# OPTIMIZED MODEL EVALUATION

print("\n--- OPTIMIZED MODEL EVALUATION ---")

best_predictions = best_model.predict(X_test)

best_mae = mean_absolute_error(y_test, best_predictions)

best_rmse = np.sqrt(
    mean_squared_error(y_test, best_predictions)
)

best_r2 = r2_score(y_test, best_predictions)

print("Optimized MAE:", best_mae)
print("Optimized RMSE:", best_rmse)
print("Optimized R2:", best_r2)


# GROUP K-FOLD CROSS VALIDATION

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

print("\n--- GROUP K-FOLD CROSS VALIDATION ---")

# Motor numaralarını eğitim verisiyle eşleştir
groups_train = data.loc[X_train.index, "unit_id"]

# Motor ID sütununu model özelliklerinden çıkar
X_train_features = X_train.copy()

# 5 katlı doğrulama
group_kfold = GroupKFold(n_splits=5)

mae_scores = []

for train_index, val_index in group_kfold.split(
    X_train_features,
    y_train,
    groups_train
):
    X_fold_train = X_train_features.iloc[train_index]
    X_fold_val = X_train_features.iloc[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_fold_train, y_fold_train)

    predictions = model.predict(X_fold_val)

    mae = mean_absolute_error(y_fold_val, predictions)

    mae_scores.append(mae)

    print("Fold MAE:", mae)

print("\nAverage Group MAE:", np.mean(mae_scores))
print("Standard Deviation:", np.std(mae_scores))


# FINAL RUL PREDICTIONS

print("\n--- FINAL RUL PREDICTIONS ---")

# Test verileriyle tahmin yap
final_predictions = gb_model.predict(X_test)

# Gerçek ve tahmin edilen değerleri karşılaştır
results = pd.DataFrame({
    "Actual RUL": y_test.values,
    "Predicted RUL": final_predictions
})

results["Prediction Error"] = (
    results["Actual RUL"] - results["Predicted RUL"]
)

print("\nFirst 10 Predictions:")
print(results.head(10).round(2))

# Tahmin sonuçlarını kaydet
results.to_csv(
    "processed/final_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")


# PREDICTION ERROR ANALYSIS

print("\n--- PREDICTION ERROR ANALYSIS ---")

# Tahmin hatası
errors = y_test.values - final_predictions

print("Mean Error:", np.mean(errors))
print("Mean Absolute Error:", np.mean(np.abs(errors)))
print("Maximum Error:", np.max(np.abs(errors)))

# Hata dağılımı grafiği
plt.figure(figsize=(10, 5))

plt.hist(errors, bins=30, edgecolor="black")

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.grid(True)
plt.savefig(
    "processed/prediction_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Error analysis completed!")
