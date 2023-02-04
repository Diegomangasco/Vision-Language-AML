import json
from collections import Counter

def getDomain(path):
    return path.split('/')[0]

def readJSON(domains):
    line = dict()
    with open("descriptions.json") as file:
        #print(file.read())
        data = json.loads(file.read())

        return {i['image_name']: i['descriptions'] for i in data if getDomain(i['image_name']) in domains}      



source_domain = 'art_painting'
target_domain = 'cartoon'
        
dict_ = readJSON([source_domain, target_domain])
print(dict_)
    