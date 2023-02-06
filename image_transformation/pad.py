from upload_image import orig_img,plot
import torchvision.transforms as T

padded_imgs = [T.Pad(padding=padding,fill=55,padding_mode="constant")(orig_img) for padding in (3, 10, 30, 50)]
# images dimensions change after padding with a 0 
plot(padded_imgs)