import  os

import numpy as np

import matplotlib

import matplotlib.pyplot as plt


attrbute0 = []
attrbute1 = []
attrbute2 = []
attrbute3 = []
attrbute4 = []
attrbute5 = []
attrbute6 = []
attrbute7 = []
attrbute8 = []
attrbute9 = []

num = 0

filename = 'magic04.txt'


with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        num += 1
        if not lines:
            break
            pass
        attr0_tmp, attr1_tmp, attr2_tmp, \
        attr3_tmp, attr4_tmp, attr5_tmp, \
        attr6_tmp, attr7_tmp, attr8_tmp, \
        attr9_tmp,attr = [i for i in lines.split(",")]
        attrbute0.append(float(attr0_tmp))
        attrbute1.append(float(attr1_tmp))
        attrbute2.append(float(attr2_tmp))
        attrbute3.append(float(attr3_tmp))
        attrbute4.append(float(attr4_tmp))
        attrbute5.append(float(attr5_tmp))
        attrbute6.append(float(attr6_tmp))
        attrbute7.append(float(attr7_tmp))
        attrbute8.append(float(attr8_tmp))
        attrbute9.append(float(attr9_tmp))
       
        pass
    pass
print sum(attrbute0)/num
print sum(attrbute1)/num
print sum(attrbute2)/num
print sum(attrbute3)/num
print sum(attrbute4)/num
print sum(attrbute5)/num
print sum(attrbute6)/num
print sum(attrbute7)/num
print sum(attrbute8)/num
print sum(attrbute9)/num


y = [attrbute0,attrbute1,attrbute2,attrbute3,attrbute4,attrbute5,attrbute6,attrbute7,attrbute8,attrbute9]
print  np.cov(y)

a=np.array(attrbute0)
b=np.array(attrbute1)
plt.plot(a,b,'ro')
#plt.show()

data = a
mean = data.mean()
std = data.std()
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
x = np.arange(-200,200,0.1)
y = normfun(x, mean, std)
plt.plot(x,y)
plt.hist(data, bins=10, rwidth=0.9, normed=True)
plt.title('Data distribution')
plt.xlabel('Data')
plt.ylabel('Probability')
#plt.show()



def variancefun(a):
    array = np.array(a)
    var = array.var()
    return var
print variancefun(attrbute0)
print variancefun(attrbute1)
print variancefun(attrbute2)
print variancefun(attrbute3)
print variancefun(attrbute4)
print variancefun(attrbute5)
print variancefun(attrbute6)
print variancefun(attrbute7)
print variancefun(attrbute8)
print variancefun(attrbute9)


def covfun(a):
    array = np.array(a)
    cov = np.cov(array)
    return cov
print covfun(attrbute0)
print covfun(attrbute1)
print covfun(attrbute2)
print covfun(attrbute3)
print covfun(attrbute4)
print covfun(attrbute5)
print covfun(attrbute6)
print covfun(attrbute7)
print covfun(attrbute8)
print covfun(attrbute9)
