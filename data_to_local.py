from PIL import Image
import deeplake
import os
import torchvision.transforms as transforms
import torch

ds = deeplake.load('hub://activeloop/wiki-art')

data_dir = "wikiart_batched_128_tensors_256"
os.makedirs(f"{data_dir}", exist_ok=True)


transform_pipeline = transforms.Compose([
    transforms.Resize((256,256) ),  # Resize all images to 128x128
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize([0.5], [0.5])
])

class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        # print(sample)
        img_data = sample["images"]  # Adjust the key based on your dataset structure
        img = Image.fromarray(img_data)  # Convert to PIL image
        return self.transform(img)

# Apply the custom transform to the DataLoader
dataloader = ds.pytorch(
    num_workers=0,
    batch_size=128,
    shuffle=True,
    transform=CustomTransform(transform_pipeline),
    # decode_method={"images": "pil"}
)

for i, sample in enumerate(dataloader):
    # print(sample.shape)
    tensor_data = sample  # Assuming the tensor is the first element in the sample
    # Save the tensor to a .pt file
    torch.save(tensor_data, f"{data_dir}/batch_{i}.pt")
    if i%100 == 0:
        print(f"Saved batch_{i}.pt")


print("All tensors saved to wikiart_tensors directory.")