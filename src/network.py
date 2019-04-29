import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import accuracy
from src.batch_norm import Batch_norm_2d

# TODO implement up_conv

class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        
        # input shape - (3,256,256)
        # output shape - (30, 256, 256)
        
        self.number_of_filters_input = 3
        self.number_of_filters_level_0 = 64
        self.number_of_filters_level_1 = 128
        self.number_of_filters_level_2 = 256
        self.number_of_filters_level_3 = 512
        self.number_of_filters_level_4 = 1024
        
        # Level 0
        
        self.level_0_conv_0 = nn.Conv2d(self.number_of_filters_input, 
                                        self.number_of_filters_level_0, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_0_conv_1 = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_filters_level_0, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
        # Level 0 -> Level 1
        
        self.pool_level_0_to_level_1 = nn.MaxPool2d(2)
        
        # Level 1
        
        self.level_1_conv_0 = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_filters_level_1, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_1_conv_1 = nn.Conv2d(self.number_of_filters_level_1, 
                                        self.number_of_filters_level_1, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
        # Level 1 -> Level 2
        
        self.pool_level_1_to_level_2 = nn.MaxPool2d(2)
        
        # Level 2
        
        self.level_2_conv_0 = nn.Conv2d(self.number_of_filters_level_1, 
                                        self.number_of_filters_level_2, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_2_conv_1 = nn.Conv2d(self.number_of_filters_level_2, 
                                        self.number_of_filters_level_2, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        # Level 2 -> Level 3
        
        self.pool_level_2_to_level_3 = nn.MaxPool2d(2)
        
        # Level 3
        
        self.level_3_conv_0 = nn.Conv2d(self.number_of_filters_level_2, 
                                        self.number_of_filters_level_3, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_3_conv_1 = nn.Conv2d(self.number_of_filters_level_3, 
                                        self.number_of_filters_level_3, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
        # Level 3 -> Level 4
        
        self.pool_level_3_to_level_4 = nn.MaxPool2d(2)
        
        # Level 4
        
        self.level_4_conv_0 = nn.Conv2d(self.number_of_filters_level_3, 
                                        self.number_of_filters_level_4, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_4_conv_1 = nn.Conv2d(self.number_of_filters_level_4, 
                                        self.number_of_filters_level_4, 
                                        3, 
                                        padding = 1, 
                                        bias=True) 
        
        # Level 4 -> Level 3
        
        self.up_conv_level_4_to_level_3 = up_conv()
        
        # Level 3
        
        self.level_3_conv_2 = nn.Conv2d(self.number_of_filters_level_4, 
                                        self.number_of_filters_level_3, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_3_conv_3 = nn.Conv2d(self.number_of_filters_level_3, 
                                        self.number_of_filters_level_3, 
                                        3, 
                                        padding = 1, 
                                        bias=True)  
        # Level 3 -> Level 2
        
        self.up_conv_level_3_to_level_2 = up_conv()
        
        # Level 2
        
        self.level_2_conv_2 = nn.Conv2d(self.number_of_filters_level_3, 
                                        self.number_of_filters_level_2, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_2_conv_3 = nn.Conv2d(self.number_of_filters_level_2, 
                                        self.number_of_filters_level_2, 
                                        3, 
                                        padding = 1, 
                                        bias=True) 
        # Level 2 -> Level 1
            
        self.up_conv_level_2_to_level_1 = up_conv()
        
        # Level 1
        
        self.level_1_conv_2 = nn.Conv2d(self.number_of_filters_level_2, 
                                        self.number_of_filters_level_1, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_1_conv_3 = nn.Conv2d(self.number_of_filters_level_1, 
                                        self.number_of_filters_level_1, 
                                        3, 
                                        padding = 1, 
                                        bias=True) 
        
        # Level 1 -> Level 0
        
        self.up_conv_level_1_to_level_0 = up_conv()
        
        # Level 0
        
        self.level_0_conv_2 = nn.Conv2d(self.number_of_filters_level_1, 
                                        self.number_of_filters_level_0, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        self.level_0_conv_3 = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_filters_level_0, 
                                        3, 
                                        padding = 1, 
                                        bias=True) 
        self.level_0_conv_4 = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_output_class, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
    def forward(self, x_level_0_left):
        # level 0
        x_level_0_left = F.relu(self.level_0_conv_0(x_level_0_left))
        x_level_0_left = F.relu(self.level_0_conv_1(x_level_0_left))
        x_level_1_left = self.pool_level_0_to_level_1(x_level_0_left)
        
        # level 1
        x_level_1_left = F.relu(self.level_1_conv_0(x_level_1_left))
        x_level_1_left = F.relu(self.level_1_conv_1(x_level_1_left))
        x_level_2_left = self.pool_level_1_to_level_2(x_level_1_left)
        
        # level 2
        x_level_2_left = F.relu(self.level_2_conv_0(x_level_2_left))
        x_level_2_left = F.relu(self.level_2_conv_1(x_level_2_left))
        x_level_3_left = self.pool_level_2_to_level_3(x_level_2_left)
        
        # level 3
        x_level_3_left = F.relu(self.level_3_conv_0(x_level_3_left))
        x_level_3_left = F.relu(self.level_3_conv_1(x_level_3_left))
        x_level_4 = self.pool_level_3_to_level_4(x_level_3_left)
        
        # level 4
        x_level_4 = F.relu(self.level_4_conv_0(x_level_4))
        x_level_4 = F.relu(self.level_4_conv_1(x_level_4))
        x_level_4 = self.up_conv_level_4_to_level_3(x_level_4)
        
        # level 3
        x_level_3_right = torch.cat((x_level_3_left, x_level_4), 1)
        x_level_3_right = F.relu(self.level_3_conv_2(x_level_3_right))
        x_level_3_right = F.relu(self.level_3_conv_3(x_level_3_right))
        
        # level 2
        x_level_2_right = torch.cat((x_level_2_left, x_level_3_right), 1)
        x_level_2_right = F.relu(self.level_2_conv_2(x_level_2_right))
        x_level_2_right = F.relu(self.level_2_conv_3(x_level_2_right))
        
        # level 1
        x_level_1_right = torch.cat((x_level_1_left, x_level_2_right), 1)
        x_level_1_right = F.relu(self.level_1_conv_2(x_level_1_right))
        x_level_1_right = F.relu(self.level_1_conv_3(x_level_1_right))
        
        # level 0
        x_level_0_right = torch.cat((x_level_0_left, x_level_1_right), 1)
        x_level_0_right = F.relu(self.level_0_conv_2(x_level_0_right))
        x_level_0_right = F.relu(self.level_0_conv_3(x_level_0_right))
        x_level_0_right = F.relu(self.level_0_conv_4(x_level_0_right))

        return x_level_0_right

def epoch_training(network, optimizer, criterion, train_set_loader, validation_set_loader):
    network.train()
    for i, data in enumerate(train_set_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    network.eval()