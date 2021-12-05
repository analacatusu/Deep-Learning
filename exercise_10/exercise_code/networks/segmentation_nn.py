"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        original_model = models.alexnet(pretrained=True)
        self.encoder = nn.Sequential(
                    # 6x6x256
                    *list(original_model.features.children())[:-3]
                )
        #self.encoder = hparams['encoder']
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(240,240), mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=23, kernel_size=1, stride=1)
        )
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
    
    def general_step(self, batch, batch_idx, mode):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images, targets = batch

        # forward pass
        predictions = self.forward(images.to(device))

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(predictions, targets.to(device))
        
        return loss

    def training_step(self, batch, batch_idx):
        loss= self.general_step(batch, batch_idx, "train")
        return {'loss': loss}
    
    def configure_optimizers(self):
        optim=torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optim
    

class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
