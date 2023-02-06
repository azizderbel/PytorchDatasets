from upload_image import orig_img,plot
import torchvision.transforms as T

center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
plot(center_crops)