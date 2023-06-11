import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.models import load_model

dataFrame = pd.read_excel("maliciousornot.xlsx")

# Veriyi İnceleme
dataFrame.info()
dataFrame.describe()
dataFrame.corr()["Type"].sort_values()
sbn.countplot(x="Type", data=dataFrame)
plt.show()
dataFrame.corr()["Type"].sort_values().plot(kind="bar")
plt.show()

# model
y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

#boyut küçültme
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#katmanlar model oluşturma
model=Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy")

#Train
model.fit(x=x_train,y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1)
#ilk epochs fazla yazıldı overfittinge yol açtı

#sonuç grafiğe dökme/Grafik Verisine Dökme
modelKaybi=pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

#katmanlar model oluşturma // Stopping
model=Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy")
earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model.fit(x=x_train,y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])

#sonuç grafiğe dökme/Grafik Verisine Dökme-2
modelKaybi=pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

#Dropout // katmanlar model oluşturma
model=Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy")

model.fit(x=x_train,y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])

#sonuç grafiğe dökme/Grafik Verisine Dökme-3
modelKaybi=pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

#tahminler
tahminler=model.predict_classes(x_test)
print(classification_report(y_test,tahminler))
print("----------------------------")
print(confusion_matrix(y_test,tahminler))