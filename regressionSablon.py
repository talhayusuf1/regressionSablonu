
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#POLİNOM DERECESİNİ ARTIRDIKCA DAHA İYİ BİR SONUC VERİR

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme(slicing)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# numpy array donusumu
X = x.values
Y = y.values

# linear regression dogrusal model olusumu
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# polynomial regression 2.dereceden
# dogrusal olmayuan 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#gorsellestirme
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()
plt.scatter(X,Y)
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)))
plt.show()

