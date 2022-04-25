import torch


root_path = "F:/datas/HFGN/"
item_path = root_path+"outfit_data.txt"
with open(item_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        break
