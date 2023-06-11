import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.models import load_model

dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(dataFrame.head()) # ilk 5 datayı getiriyo başını gösteriyo
sbn.pairplot(dataFrame)
plt.show()

#veriyi test/train olaral ikiye ayırmak
#y=wx+b      y->label        x->feature(özellik)
#test bölme         train_test_split()
y=dataFrame["Fiyat"].values
x=dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values     #values denmezse pandas olur derse numpy dizisi olur
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)
print(x_train)
print(x_train.shape)
print(x_test.shape)

#scaling => boyutunu değiştirmek      MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test) # 0 ile 1 arasına getirildi

#model hazırlama
model=Sequential()
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="rmsprop",loss="mse") #rmsprop ve adam optimazson algoritmaları

#eğitim
model.fit(x_train,y_train,epochs=250)
loss=model.history.history["loss"]
sbn.lineplot(x=range(len(loss)),y=loss)
plt.show()
trainLoss=model.evaluate(x_train,y_train,verbose=0)
testLoss=model.evaluate(x_test,y_test,verbose=0)
print("Train: ",trainLoss,"\n ","Test: ", testLoss)

#Tahminler

testTahminleri=model.predict(x_test)
tahminDF=pd.DataFrame(y_test,columns=["Gercek Y"])
testTahminleri=pd.Series(testTahminleri.reshape(330,))
tahminDF=pd.concat([tahminDF,testTahminleri],axis=1)
tahminDF.columns=["Gercek Y", "Tahmin Y"]
print(tahminDF)
sbn.scatterplot(x="Gercek Y", y= "Tahmin Y", data=tahminDF)
plt.show()

#sapma
mean_absolute_error(tahminDF["Gercek Y"],tahminDF["Tahmin Y"])
mean_squared_error(tahminDF["Gercek Y"],tahminDF["Tahmin Y"])
print(mean_absolute_error(tahminDF["Gercek Y"],tahminDF["Tahmin Y"]))
print(dataFrame.describe())

#yeni bisiklet verisi
yeniBisiklerOzellikleri=[[1753,1751]]
yeniBisiklerOzellikleri=scaler.transform(yeniBisiklerOzellikleri)
print(model.predict(yeniBisiklerOzellikleri))

#kayıt etmek
model.save("bisiklet_modeli.h5")
sonradanCagirilanModel=load_model("bisiklet_modeli.h5")
sonradanCagirilanModel.predict(yeniBisiklerOzellikleri)