import requests
path = "C:/Users/12578/Desktop/item_data.txt"
save_path = "F:/datas/"
with open(path, 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        url = line.split(',')[2]
        r = requests.get(url)
        if r.status_code == 200:  # 返回是200，说明请求成功
            img_path = save_path + url.split('!')[-1]  # 图片保存路径
            with open(img_path, 'wb') as ff:
                ff.write(r.content)
            print(img_path, "is saved")
        else:
            print(url, "save fail !!!!!!!!!!!")
