from torch import nn


class PDN(nn.Module):
    """
    Patch Description Network (PDN) model with configurable size ('small' or 'medium') and output channels.
    
    Args:
        size (str): Model size, either 'small' or 'medium'. Defaults to 'small'.
        out_channels (int): Number of output channels for the final convolutional layer. Defaults to 384.
        padding (bool): Whether to apply padding in convolutional layers. Defaults to False.
    """
    def __init__(self, size='small', out_channels=384, padding=False):
        super(PDN, self).__init__()
        
        # Determine padding multiplier
        pad_mult = 1 if padding else 0
        
        if size == 'medium':
            # Medium-sized architecture
            self.conv0 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3 * pad_mult)
            self.relu0 = nn.ReLU(inplace=True)
            self.pool0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
            
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3 * pad_mult)
            self.relu1 = nn.ReLU(inplace=True)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
            
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
            self.relu2 = nn.ReLU(inplace=True)
            
            self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1 * pad_mult)
            self.relu3 = nn.ReLU(inplace=True)
            
            self.conv4 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4)
            self.relu4 = nn.ReLU(inplace=True)
            
            self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            
        else:
            # Small-sized architecture
            self.conv0 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
            self.relu0 = nn.ReLU()
            self.pool0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
            
            self.conv1 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
            
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
            self.relu2 = nn.ReLU()
            
            self.conv3 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1)

    def forward(self, x):
        """
        Forward pass of the PDN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        if hasattr(self, 'conv5'):  
            # Medium-sized architecture
            x = self.conv0(x)
            x = self.relu0(x)
            x = self.pool0(x)

            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)

            x = self.conv3(x)
            x = self.relu3(x)

            x = self.conv4(x)
            x = self.relu4(x)

            x = self.conv5(x)

        else:  
            # Small-sized architecture
            x = self.conv0(x)
            x = self.relu0(x)
            x = self.pool0(x)

            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)

            x = self.conv3(x)
        return x