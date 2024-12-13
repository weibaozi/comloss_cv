import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger


# Define the model using PyTorch Lightning
class EfficientNetBinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=1, lr=1e-3):
        super(EfficientNetBinaryClassifier, self).__init__()
        self.save_hyperparameters()

        # Load EfficientNet B0 pre-trained on ImageNet
        # self.model = models.efficientnet_b0(pretrained=True)
        self.model = models.efficientnet_b7(pretrained=True)
        
        # Replace the classifier head
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # Define binary cross-entropy loss and accuracy metric
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()  # Reshape for binary classification
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.sigmoid(), y.int())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1).float()  # Reshape for binary classification
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.sigmoid(), y.int())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# Data transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# datasets= datasets.ImageFolder(root='imgs', transform=data_transforms['train'])
# print(datasets)
# # Print the class-to-index mapping
# print(f"Class-to-index mapping: {datasets.class_to_idx}")

# # Print the total number of images
# print(f"Total number of images: {len(datasets)}")
# # split dataset to train and test
# train_size = int(0.8 * len(datasets))
# test_size = len(datasets) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
# #data loader
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4)
# # Datasets and dataloaders
# Set up a TensorBoard logger
if __name__ == '__main__':
    
    logger = TensorBoardLogger("tb_logs", name="efficientnet_b7")
    train_dataset = datasets.ImageFolder(root='img2/train', transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root='img2/test', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Instantiate the PyTorch Lightning model
    model = EfficientNetBinaryClassifier()

    # Set up the Trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=30,
        accelerator='gpu',  # Use GPU if available
        devices=1

    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    #save
    torch.save(model.state_dict(), 'modeleb7.pth')
