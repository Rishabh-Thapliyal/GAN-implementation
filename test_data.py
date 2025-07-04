import deeplake
import torchvision.transforms as transforms
ds = deeplake.load('hub://activeloop/wiki-art')

# Define a transform pipeline to resize and convert images to tensors
transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),         # Convert images to tensors
])

# Custom transform to apply the pipeline to each sample
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        print(sample)
        img_data = sample[0][1]  # Adjust the key based on your dataset structure
        img = Image.fromarray(img_data)  # Convert to PIL image
        return self.transform(img)

# Apply the custom transform to the DataLoader
dataloader = ds.pytorch(
    num_workers=2,
    batch_size=32,
    shuffle=True,
    transform=CustomTransform(transform_pipeline)
)

for sample in ds:
    print(type(sample))
    break

for epoch in range(2):
    for i, (imgs, _) in enumerate(dataloader):
        print(imgs)
        print(type(imgs))
        print(type(_))
        break