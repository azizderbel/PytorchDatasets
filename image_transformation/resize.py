from upload_image import orig_img,plot
import torchvision.transforms as T

resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
plot(resized_imgs)