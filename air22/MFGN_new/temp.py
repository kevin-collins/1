import json
import random


path1 = "F:\datas\polyvore_outfits\disjoint/val_posi.txt"
path2 = "F:\datas\polyvore_outfits\disjoint/val_posi"
save_path = "F:\datas\polyvore_outfits\disjoint/compatibility_test_easy.json"

o = []
items = []
with open(path, 'r') as f:
    datas = json.load(f)
    for outfit in datas:
        if outfit['label'] == '1':
            o.append(outfit)
            for i in outfit['items']:
                items.append(i)
print(len(items))
with open(path, 'r') as f:
    datas = json.load(f)
    for outfit in datas:
        if outfit['label'] == '1':
            p = int(random.randint(0, len(outfit['items'])-1))
            outfit['label'] = '0'
            outfit['items'][p] = items[random.randint(0, len(items)-1)]
            o.append(outfit)

with open(save_path, 'w') as f:
    json.dump(o, f)
exit(0)


path = "F:\datas\polyvore_outfits\disjoint/compatibility_test_new.json"
save_path = "F:\datas\polyvore_outfits\disjoint/compatibility_test_easy.json"

o = []
items = []
with open(path, 'r') as f:
    datas = json.load(f)
    for outfit in datas:
        if outfit['label'] == '1':
            o.append(outfit)
            for i in outfit['items']:
                items.append(i)
print(len(items))
with open(path, 'r') as f:
    datas = json.load(f)
    for outfit in datas:
        if outfit['label'] == '1':
            p = int(random.randint(0, len(outfit['items'])-1))
            outfit['label'] = '0'
            outfit['items'][p] = items[random.randint(0, len(items)-1)]
            o.append(outfit)

with open(save_path, 'w') as f:
    json.dump(o, f)
exit(0)


path = "F:\datas\polyvore_outfits\disjoint/compatibility_test_new.json"
lines_pos = []
lines_nega = []
with open(path, 'r') as f:
    datas = json.load(f)

    for outfit in datas:
        line = ""
        label = outfit['label']
        for i in outfit['items']:
            line = line + i['im']
            if i!=outfit['items'][-1]:
                line += " "
        if label == '1':
            lines_pos.append(line+'\n')
        else:
            lines_nega.append(line + '\n')

with open("F:\projects\MFGN/train_posi_repalceone.txt", 'w') as f:
    for line in lines_pos:
        f.writelines(line)


with open("F:\projects\MFGN/train_nega_repalceone.txt", 'w') as f:
    for line in lines_nega:
        f.writelines(line)

exit(0)