from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/bees_image/92663402_37f379e57a.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 3, dataformats="HWC")
# y = x
for i in range(100):
    writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)

writer.close()
