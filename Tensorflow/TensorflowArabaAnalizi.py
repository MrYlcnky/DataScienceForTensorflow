import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error


# exceli okuma
dataFrame = pd.read_excel("merc.xlsx")
print(dataFrame.head())

# veriyi anlama      dataFrame.describe()
print(dataFrame.describe())
print(dataFrame.isnull().sum(), " isnull")  # Boş veri var mı? varsa kaç tane

# Grafik Analiz
plt.figure(figsize=(7, 5))
sbn.distplot(dataFrame["price"])  # distplot
plt.show()
sbn.countplot(x=dataFrame["year"])
plt.show()
dataFrame.drop("transmission", axis=1, inplace=True)
print(dataFrame.corr())  # ne neyle iligili olan
print(dataFrame.corr()["price"].sort_values())  # paraya göre pozitif negatif etkileyeni sıralıyo

# transmission çıkarıldı str içerdiği için etkilemediği için fiyatını ne belirlediği poztif negatife bakıldı

# fiyatı yüksek arabalar
sbn.scatterplot(x="mileage", y="price", data=dataFrame)
plt.show()
print(dataFrame.sort_values("price", ascending=False).head(20))
print(dataFrame.sort_values("price", ascending=True).head(20))
len(dataFrame) * 0.01  # kadar en yüksek fiyatlı arabaları çıkaralım

# Az veri olarak yüksek fiyatlı arabalar olduğu tespıt edildi

# Veri Temizliği
yuzdeDoksanDokuzDF = pd.DataFrame(dataFrame.sort_values("price", ascending=False).iloc[131:])
print(yuzdeDoksanDokuzDF,"99*")
print(dataFrame.groupby("year").mean()["price"])
print(dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"])
dataFrame = yuzdeDoksanDokuzDF
print(dataFrame)
dataFrame = dataFrame[dataFrame.year != 1970]

# yüksek fiyatlı arabaları çıkardık ama veriden %1 lik oranda

# Model Oluşturmak
y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

#boyut
scaler =MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#katmanlar model oluşturma
model=Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

#eğitme
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)
#validation_data:otomize yapma işlemi yapar

#Grafik Verisine Dökme
kayipVerisi=pd.DataFrame(model.history.history)
kayipVerisi.head()
kayipVerisi.plot()
plt.show()

#Tahmin Sonuç Değerlendirme
tahminDizisi=model.predict(x_test)
print(mean_absolute_error(y_test,tahminDizisi))
plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")
plt.show()

#veri setinden tahmin
yeniArabaSeries=dataFrame.drop("price",axis=1).iloc[2]
yeniArabaSeries=scaler.transform(yeniArabaSeries.values.reshape(-1,5))
print(model.predict(yeniArabaSeries))