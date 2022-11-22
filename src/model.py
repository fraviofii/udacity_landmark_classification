import torch
import torch.nn as nn

# define the CNN architecture
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
#         super().__init__()
#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))
        
#         self.model = nn.Sequential(
#             # First conv + maxpool + relu (3x224x224)
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2, 2),
#             #nn.Dropout2d(0.2),
            
#             # Second conv + maxpool + relu (16x112X112)
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, 2),
#             #nn.Dropout2d(0.2),
            
#             # Third conv + maxpool + relu (32x56x56)
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(),
#             #nn.BatchNorm2d(64),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout2d(0.2),
            
             
#             # Fourth conv + maxpool + relu (64x28x28)
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.2),
            
#             # Flatten feature maps (128x14x14)
#             nn.Flatten(),
            
#             # Fully connected layers. This assumes that the input image was 28x28
#             nn.Linear(512*7*7, 256),
#             #nn.BatchNorm1d(1024),
#             nn.ReLU(),
            
#             #nn.Linear(1024, 512),
#             #nn.BatchNorm1d(512),
#             #nn.ReLU(),
#             nn.Dropout(p=dropout),
            
#             nn.Linear(256, num_classes)
#     )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         return self.model(x)

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            # convolutional layer 1. It sees 3x224x224 image tensor
            # and produces 16 feature maps 224x224 (i.e., a tensor 16x224x224)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            # 2x2 pooling with stride 2. It sees tensors 16x224x224
            # and halves their size, i.e., the output will be 16x112x112
            nn.MaxPool2d(2, 2),
            
            # convolutional layer (sees the output of the prev layer, i.e.,
            # 16x112x112 tensor)
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            nn.BatchNorm2d(32),
            
            # convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            
            nn.Dropout(dropout),
            
            # linear layer (128 * 14 * 14 -> 500)
            nn.Flatten(),  # 1x64x28x28 
            
            nn.Linear(64 * 28 * 28, 500),  # -> 500 # 3 layers
            nn.BatchNorm1d(500),
            nn.Dropout(dropout),

            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)

# define the CNN architecture
class MyModel5(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            # convolutional layer 1. It sees 3x224x224 image tensor
            # and produces 16 feature maps 224x224 (i.e., a tensor 16x224x224)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 2x2 pooling with stride 2. It sees tensors 16x224x224
            # and halves their size, i.e., the output will be 16x112x112
            nn.MaxPool2d(2, 2),
            
            # convolutional layer (sees the output of the prev layer, i.e.,
            # 16x112x112 tensor)
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            # convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            
            # convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14

            nn.BatchNorm2d(128),

            # convolutional layer
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256x7x7
            
            nn.Dropout(dropout),
            
            # linear layer (128 * 14 * 14 -> 500)
            nn.Flatten(),  # -> 1x256x7x7 -> 1x64x28x28 # -> 1x128x14x14
            
            #nn.Linear(128 * 14 * 14, 500),  # -> 500 # 4 layers
            #nn.Linear(64 * 28 * 28, 500),  # -> 500 # 3 layers
            nn.Linear(256 * 7 * 7, 500), # -> 500 # 5 layers
            nn.Dropout(dropout),

            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
