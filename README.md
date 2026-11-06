# Predictive Maintenance & RUL Prediction

Bu proje, motor sensör verilerini kullanarak bir motorun **Remaining Useful Life (RUL)** değerini tahmin etmek amacıyla geliştirilmiştir.

## Projenin Amacı

Amaç, motorun çalışma döngüsü ve sensör verilerini kullanarak motorun yaklaşık kaç çalışma döngüsü daha kullanılabileceğini tahmin etmektir.

Proje kapsamında veri analizi, makine öğrenmesi modeli karşılaştırması, model değerlendirmesi, özellik önem analizi ve Streamlit tabanlı tahmin uygulaması gerçekleştirilmiştir.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Kullanılan Modeller

Projede farklı makine öğrenmesi yaklaşımları karşılaştırılmıştır:

| Model             |   MAE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: |
| Gradient Boosting | 22.62 | 29.59 | 0.797 |
| Random Forest     | 23.89 | 31.50 | 0.770 |
| Sensor Only       | 29.89 | 41.33 | 0.626 |

En iyi sonucu **Gradient Boosting** modeli vermiştir.

## Model Optimizasyonu

Gradient Boosting modeli için GridSearchCV kullanılarak farklı parametreler test edilmiştir.

Kullanılan parametreler:

- `learning_rate = 0.03`
- `max_depth = 4`
- `n_estimators = 150`

## Cross Validation

Model değerlendirmesinde motorların `unit_id` değerleri kullanılarak **GroupKFold** uygulanmıştır.

Bu yöntem, aynı motora ait verilerin hem eğitim hem de doğrulama grubunda bulunmasını engelleyerek veri sızıntısı riskini azaltır.

GroupKFold sonucunda ortalama MAE yaklaşık **27.72** olarak elde edilmiştir.

## Feature Importance

Modelin tahminlerinde en etkili özelliklerden biri `cycle` değişkenidir.

Bunun yanında bazı sensör değişkenleri de model açısından önemli bulunmuştur:

- sensor_11
- sensor_4
- sensor_12
- sensor_7
- sensor_9

Feature importance değerleri, modelin tahmin yaparken hangi değişkenlerden daha fazla yararlandığını gösterir.

## Streamlit Uygulaması

Projede modelin kullanılabileceği basit bir **Streamlit web uygulaması** geliştirilmiştir.

Kullanıcı:

- Cycle
- Operating settings
- Sensor değerleri

gibi bilgileri girerek motorun tahmini kalan ömrünü görebilir.

Uygulama ayrıca model performansını ve feature importance sonuçlarını göstermektedir.

Uygulamayı çalıştırmak için:

`streamlit run app.py`

## Proje Yapısı

```text
defense-maintenance-project/
│
├── data/
├── raw/
├── processed/
├── main.py
├── app.py
├── gradient_boosting_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Kurulum

Gerekli Python kütüphanelerini yüklemek için:

`pip install -r requirements.txt`

Makine öğrenmesi kodunu çalıştırmak için:

`python main.py`

Streamlit uygulamasını başlatmak için:

`streamlit run app.py`

## Sonuç

Gradient Boosting modeli test verisi üzerinde yaklaşık **0.797 R²** değerine ulaşmıştır.

Bu proje, sensör verilerinden yararlanarak makine öğrenmesi ile **predictive maintenance** ve **Remaining Useful Life prediction** yaklaşımının nasıl uygulanabileceğini göstermektedir.
