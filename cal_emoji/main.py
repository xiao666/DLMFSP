import pandas as pd 
import numpy as np
import csv

#data = pd.read_csv('t_prob64.csv')#sum t_prob
#data = pd.read_csv('top5_emoji.csv')#sum top_5 occur times
data = pd.read_csv('selected_top5_emoji.csv')#sum top_5 occur times

print (type(data))
'''
data2=data.values
data2=np.asmatrix(data2)
print (type(data2))
print (len(data2))
print (data2[:,0])
print (data2[0,1])#1row, 2col
'''

aggregated_emoji=[[0 for x in range(20)] for y in range(1989)]

print ("len(data.index):",len(data.index))#49725
#print (sum(data.iloc[25*0:25*0+25,0]))
#len=len(data.index)/5
for j in range(20):#64
    temp=[]
    for i in range(1989):
        aggregated_emoji[i][j]=sum(data.iloc[25*i:25*i+25,j])


#print (aggregated_emoji[0])
#print (aggregated_emoji[1])

#######################################remember change output file name
'''
OUTPUT_PATH='sum_top5.csv'
print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Emoji_0', 'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4',
                         'Emoji_5', 'Emoji_6', 'Emoji_7', 'Emoji_8', 'Emoji_9', 'Emoji_10',
                         'Emoji_11', 'Emoji_12', 'Emoji_13', 'Emoji_14', 'Emoji_15',
                         'Emoji_16', 'Emoji_17', 'Emoji_18', 'Emoji_19', 'Emoji_20',
                         'Emoji_21', 'Emoji_22', 'Emoji_23', 'Emoji_24', 'Emoji_25',
                         'Emoji_26', 'Emoji_27', 'Emoji_28', 'Emoji_29', 'Emoji_30',
                         'Emoji_31', 'Emoji_32', 'Emoji_33', 'Emoji_34', 'Emoji_35',
                         'Emoji_36', 'Emoji_37', 'Emoji_38', 'Emoji_39', 'Emoji_40',
                         'Emoji_41', 'Emoji_42', 'Emoji_43', 'Emoji_44', 'Emoji_45',
                         'Emoji_46', 'Emoji_47', 'Emoji_48', 'Emoji_49', 'Emoji_50',
                         'Emoji_51', 'Emoji_52', 'Emoji_53', 'Emoji_54', 'Emoji_55',
                         'Emoji_56', 'Emoji_57', 'Emoji_58', 'Emoji_59', 'Emoji_60',
                         'Emoji_61', 'Emoji_62', 'Emoji_63'])#in all 64 emojis
    for i, row in enumerate(aggregated_emoji):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
'''


OUTPUT_PATH='selected_sum_top5.csv'
print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow([ 'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4',
                         'Emoji_5', 'Emoji_6', 'Emoji_7', 'Emoji_8', 'Emoji_9', 'Emoji_10',
                         'Emoji_11', 'Emoji_12', 'Emoji_13', 'Emoji_14', 'Emoji_15',
                         'Emoji_16', 'Emoji_17', 'Emoji_18', 'Emoji_19', 'Emoji_20'
                    ])#in all 20 emojis
    for i, row in enumerate(aggregated_emoji):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
