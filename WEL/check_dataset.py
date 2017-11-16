
import numpy as np
import os
import pandas as pd
from collections import Counter



###LSTM###
#import data
data = pd.read_csv('Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

train_y=np.array(train["Label"])
test_y=np.array(test["Label"])

c1=Counter(train_y)
print (c1)
c2=Counter(test_y)
print (c2)
