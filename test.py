import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from time import time
from mobilenet_v2 import EfficientNetBinaryClassifier  # Ensure you have this model script available

# Set device to CPU for inference
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# Define data transformations (same as used during training)
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the validation dataset
val_dataset = datasets.ImageFolder(root='img2/test', transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # Single image per batch for testing

# Load the trained model
checkpoint = torch.load('model.pth', map_location=device)
print("loaded model")
model = EfficientNetBinaryClassifier()
model.load_state_dict(checkpoint)
# model.to(device)  # Load the model onto the CPU
model.eval()  # Set model to evaluation mode
model.to(device)


# Run inference on a few images and record time
num_images = 10  # Number of images to run inference on
times = []  # List to store inference times
outputs = []  # Store model outputs for each image

with torch.no_grad():  # Disable gradient computation for inference
    for i, (image, label) in enumerate(val_loader):
        if i >= num_images:  # Stop after processing `num_images`
            break
        
        image = image.to(device)
        
        # Record start time
        start_time = time()
        
        # Run inference
        output = model(image)
        
        # Record end time
        end_time = time()
        
        # Store inference time
        times.append(end_time - start_time)
        
        # Store the model's output (sigmoid for binary classification)
        outputs.append(output.sigmoid().cpu().numpy())
        
        print(f"Image {i + 1}: Inference time: {times[-1]:.4f} seconds, Output: {outputs[-1]},label: {label}")
        
# Calculate average time for inference
average_time = sum(times) / len(times)
print(f"\nAverage inference time per image: {average_time:.4f} seconds")

# Optionally, you can write the outputs and times to a file if needed
