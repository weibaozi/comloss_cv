import torch
from torchvision import transforms
from PIL import Image
from mobilenet_v2 import EfficientNetBinaryClassifier
import cv2
import numpy as np

class myModel:
    def __init__(self, model_path, device):
        """
        Initialize the inference class.
        
        Args:
            model_path (str): Path to the saved model checkpoint.
            device (torch.device): The device to load the model onto.
        """
        self.device = device
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),  # Convert OpenCV image (numpy array) to PIL format
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print("Loaded model.")
        
        # Initialize and load the model
        self.model = EfficientNetBinaryClassifier()
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, cv_image):
        """
        Preprocess the input image in OpenCV format.
        
        Args:
            cv_image (numpy.ndarray): The input image in OpenCV (numpy) format.
        
        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        if not isinstance(cv_image, np.ndarray):
            raise ValueError("Input image must be a numpy array in OpenCV format.")

        # Convert BGR (OpenCV) to RGB (PIL expects RGB format)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        image_tensor = self.data_transforms(cv_image)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def predict(self, cv_image):
        """
        Make a prediction for a given OpenCV image.
        
        Args:
            cv_image (numpy.ndarray): The input image in OpenCV (numpy) format.
        
        Returns:
            torch.Tensor: The prediction logits or probabilities.
        """
        input_tensor = self.preprocess_image(cv_image).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
