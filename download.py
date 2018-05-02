import os


classnames = [
    "trumpet",
    "trombone",
    "harp",
    "guitar",
    "violin",
    "cow",
    "cat",
    "dog",
    "clarinet",
    "piano",
    "zigzag",
    "duck",
]

urls = ['https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy'.format(name)
    for name in classnames]

# urls = [
#     'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/dog.npy', 
#     'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy', 
#     'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/mouse.npy', 
#     'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/pig.npy', 
#     'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/rabbit.npy'
# ]

def download(url):
    filename = os.path.basename(url)
    path = './data/%s' % filename
    dst = os.path.join('data', filename)

    if os.path.exists(dst):
        print(path, 'exists')
        return

    if not os.path.exists(path):
        ret = os.system('wget "%s" -O %s' % (url, path))
        if ret == 0: # ダウンロード成功したらリンク
            os.system('ln -s %s %s' % ('../' + path, dst))
        else: # ダウンロード失敗したらお片付け
            os.system('rm %s' % path)
    else:
        print(path, 'exists')
        os.system('ln -s %s %s' % ('../' + path, dst))

for url in urls:
    download(url)
