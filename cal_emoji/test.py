import numpy as np


A=[1,2,3]
B=[4,5,6]
C=[7,8,9]



D=[]
D.append(A)
D.append(B)
D.append(C)
D.append(A)

print (D)

D=np.reshape(D,(2,6))
print (D)