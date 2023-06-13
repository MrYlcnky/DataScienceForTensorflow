import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

#ExcelOkuma
dataFrame=pd.read_csv("charges.csv")

#VeriInceleme
print(dataFrame.describe())
print(dataFrame.head())
print(dataFrame.isnull().sum())
print(dataFrame["region"].unique())
print(dataFrame["region"].nunique())
print(dataFrame["region"].value_counts())
print(dataFrame["smoker"].unique())

#StrVerileriniDonusturme
def encode(x):
    if x == "yes":
        return 1
    elif x =="no":
        return 0
dataFrame["NumSmoker"]=dataFrame.apply(lambda x: encode(x["smoker"]),axis=1)

def encode(y):
    if y == "southwest":
        return 1
    elif y =="southeast":
        return 2
    elif y =="northwest":
        return 3
    elif y =="northeast":
        return 4
dataFrame["NumRegion"]=dataFrame.apply(lambda y: encode(y["region"]),axis=1)

def encode(z):
    if z == "female":
        return 1
    elif z =="male":
        return 2
dataFrame["NumSex"]=dataFrame.apply(lambda z: encode(z["sex"]),axis=1)

#StringleriKaldırma
print(dataFrame)
dataFrame.drop("sex",axis=1,inplace=True)
dataFrame.drop("region",axis=1,inplace=True)
dataFrame.drop("smoker",axis=1,inplace=True)
print(dataFrame.corr()["charges"])

#GrafikAnaliz
sbn.barplot(dataFrame)
plt.show()
sbn.distplot(x=dataFrame["charges"])
plt.show()
sbn.histplot(x=dataFrame["charges"])
plt.show()
sbn.countplot(x=dataFrame["charges"])
plt.show()
sbn.boxplot(dataFrame)
plt.show()

#ModelOluşturma
y=dataFrame["charges"].values
x=dataFrame.drop("charges",axis=1).values
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
model.fit(x=x_train,y=y_train, epochs=400,validation_data=(x_test,y_test),verbose=1,batch_size=1)

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

#DatasettenTahmin
chargesSeries=dataFrame.drop("charges",axis=1).iloc[2]
chargesSeries=scaler.transform(chargesSeries.values.reshape(-1,6))
print(model.predict(chargesSeries))