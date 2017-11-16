import json

with open('Microblog_Trialdata.json','r') as f:
    data=json.load(f)


print (type(data))
print (data)
