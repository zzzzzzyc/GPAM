from PIL import Image

path = '/home/hpclp/disk/Graphgpt/InternVL/internvl_chat/shell/data/MR2/train/img/3411.jpg'
try:
    img = Image.open(path)
    img.verify()
    print("✅ 图像可用")
except Exception as e:
    print(f"❌ 图像打开失败: {type(e)} - {e}")
