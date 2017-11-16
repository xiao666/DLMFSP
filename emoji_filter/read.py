#read from json file, store in csv file

#import simplejson
import json
import csv

with open('Headline_Trainingdata.json','r') as f2:
    data=json.load(f2)

'''

print (data[0]['id'])


#print (data['id'])
csv_text=[[0 for x in range(3)] for y in range(len(data))]
for i in range(len(data)):
    csv_text[i][0]=data[i]['title']
    csv_text[i][1]=data[i]['sentiment']
    if(float(data[i]['sentiment'])>=0):
        #"1" means bullish; "0" means bearish
        csv_text[i][2]=1
    else:
        csv_text[i][2]=0


#write into csv file
OUTPUT_PATH='semeval_2.csv'
print('Writing results to {}'.format(OUTPUT_PATH))
with open (OUTPUT_PATH,'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['text', 'sentiment_score', 'binary_sentiment'])#in all 64 emojis
    for i, row in enumerate(csv_text):
        try:
            writer.writerow(row)
        except:
            print("Exception at row {}!".format(i))
'''