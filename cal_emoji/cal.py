import pandas as pd
import numpy as np
import csv

data = pd.read_csv('t_prob64.csv')
data=np.array(data)
#sum1=sum(data[0]) #each row = 1.0
idx=[0,4,7,10,16,31,36,53,57,62,
     2,3,5,27,29,34,39,43,46,52]

data=data[:,idx]

largest=[[0 for x in range(20)] for y in range(len(data))]
top5_idx=[]

for i in range(len(data)):
    temp=data[i]
    temp_idx=[]
    z_idx=(-temp).argsort()[:5]#largest k=5 number

    top5_idx.append(z_idx)
    for h in z_idx:
        largest[i][h]=1

top5_idx=np.reshape(top5_idx,(1989,125))
print (top5_idx[0][5:10])


'''

OUTPUT_PATH='top5_emoji.csv'
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
    for i, row in enumerate(largest):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))

'''
'''
OUTPUT_PATH='top5_index+1.csv'
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
                         'Emoji_61', 'Emoji_62', 'Emoji_63','x1',
                         'Emoji_0', 'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4',
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
                         'Emoji_56', 'Emoji_57', 'Emoji_58', 'Emoji_59'

                         ])#in all 64 emojis
    for i, row in enumerate(top5_idx):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
'''


OUTPUT_PATH='selected_top5_emoji.csv'
print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4',
                         'Emoji_5', 'Emoji_6', 'Emoji_7', 'Emoji_8', 'Emoji_9', 'Emoji_10',
                         'Emoji_11', 'Emoji_12', 'Emoji_13', 'Emoji_14', 'Emoji_15',
                         'Emoji_16', 'Emoji_17', 'Emoji_18', 'Emoji_19', 'Emoji_20'

                         ])#in all 64 emojis
    for i, row in enumerate(largest):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
