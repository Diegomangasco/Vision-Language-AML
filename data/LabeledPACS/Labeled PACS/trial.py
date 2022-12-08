import json
from collections import Counter

example = None
with open("descriptions.json") as file:
    #print(file.read())
    data = json.loads(file.read())
    keys = [i['image_name'] for i in data]
    s = set(keys)
    if len(keys) == len(s):
        print('Bravo')
    else:
        print(f'{len(s)}, {len(keys)}, Cattivo')
        count = dict(Counter(keys))
        for c, v in count.items():
            if v == 2:
                print(c)