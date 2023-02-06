from upload_image import orig_img,plot
import torchvision.transforms as T

policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
augmenters = [T.AutoAugment(policy) for policy in policies]

imgs = [
    [augmenter(orig_img) for _ in range(4)]
    for augmenter in augmenters
]

row_title = [str(policy).split('.')[-1] for policy in policies]
plot(imgs, row_title=row_title)