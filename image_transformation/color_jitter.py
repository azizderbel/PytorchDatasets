from upload_image import orig_img,plot
import torchvision.transforms as T

# Random transformation
# means that the same transfomer instance will 
# produce different result each time it transforms a given image

jitter = T.ColorJitter(brightness=.5, hue=.3)
jitted_imgs = [jitter(orig_img) for _ in range(4)]
plot(jitted_imgs)