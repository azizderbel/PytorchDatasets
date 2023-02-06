from upload_image import orig_img,plot
import torchvision.transforms as T

augmenter = T.RandAugment()
imgs = [augmenter(orig_img) for _ in range(4)]
plot(imgs)