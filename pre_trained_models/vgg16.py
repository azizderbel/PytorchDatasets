import torch
from torchvision.io import read_image
from torchvision.models import vgg16,VGG16_Weights


# Step 1: Initialize model with the best available weights
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
img = read_image("pre_trained_models/dog.jpg")
preprocess = weights.transforms()

# Step 4: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 3: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()

category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")