from upload_image import orig_img,plot
import torchvision.transforms as T

# Random transformation
elastic_transformer = T.ElasticTransform(alpha=250.0)
transformed_imgs = [elastic_transformer(orig_img) for _ in range(2)]
plot(transformed_imgs)