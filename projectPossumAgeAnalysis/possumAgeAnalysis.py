import tensorflow as tf
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

#Excelokuma
dataFrame=pd.read_csv("possum.csv")
print(dataFrame.head())

#Analiz

print(dataFrame.isnull().sum())
dataFrame=dataFrame.dropna()
print(dataFrame.isnull().sum())
print(dataFrame.describe())

#str ve gereksiz attribute'ları atalım
print(dataFrame["Pop"].unique()) #vic,other
print(dataFrame["sex"].unique()) #f m

def encode(x):
    if x == "Vic":
        return 1
    elif x == "other":
        return 0
dataFrame["NumPop"]=dataFrame.apply(lambda x: encode(x["Pop"]),axis=1)
def encode(y):
    if y == "f":
        return 1
    elif y == "m":
        return 0
dataFrame["NumSex"]=dataFrame.apply(lambda y: encode(y["sex"]),axis=1)

dataFrame.drop("case",axis=1,inplace=True)
dataFrame.drop("Pop",axis=1,inplace=True)
dataFrame.drop("sex",axis=1,inplace=True)

print(dataFrame.corr()["age"])

#GrafikAnaliz
plt.figure(figsize=(15,10))
sbn.pairplot(dataFrame)
plt.show()
sbn.barplot(dataFrame)
plt.show()
sbn.histplot(dataFrame)
plt.show()
sbn.distplot(dataFrame["age"])
plt.show()

#ModelOluşturma
y=dataFrame["age"].values
x=dataFrame.drop("age",axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

#BoyutAyarlama
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#KatmnalarModelOluşturma
model=Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")


#Train
model.fit(x=x_train,y=y_train, epochs=120,validation_data=(x_test,y_test),verbose=1,batch_size=1)

#Grafik Verisine Dökme
kayipVerisi=pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
plt.show()

#Tahmin Sonuç Değerlendirme
tahminDizisi=model.predict(x_test)
print(mean_absolute_error(y_test,tahminDizisi))
plt.scatter(x=y_test,y=tahminDizisi)
plt.plot(y_test,y_test,"r*-")
plt.show()

#veri setinden tahmin
possumSeries=dataFrame.drop("age",axis=1).iloc[2]
possumSeries=scaler.transform(possumSeries.values.reshape(-1,12))
print(model.predict(possumSeries))
