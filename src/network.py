import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import datetime
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils import flatten
from src.visualization import ids_to_rgb_image_pytorch
from src.data_loading import id_to_colour_dict

class Conv_series(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3], dropout_prob = 0):
        super().__init__()
        filters = [number_of_filters_input] + [number_of_filters_output]*len(kernels)
        layers =  flatten([[nn.Conv2d(f1, f2, k, padding=k // 2, bias=False),
                            nn.BatchNorm2d(f2), 
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(p=dropout_prob)] 
                            for k, f1, f2 in zip(kernels, filters, filters[1:])])
        
        self.bottom = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.bottom(x)
        return x

class Down_step(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3], dropout_prob = 0):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_series(number_of_filters_input, number_of_filters_output, kernels, dropout_prob)
        )
        
    def forward(self, x):
        x = self.down(x)
        return x
    
class Up_step(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3]):
        super().__init__()
        self.up = nn.ConvTranspose2d(number_of_filters_input, number_of_filters_output, 2, stride = 2)
        self.conv_series = Conv_series(2*number_of_filters_output, number_of_filters_output, kernels)
        
    def forward(self, x, residue):
        x = self.up(x)
        x = torch.cat((residue, x), 1)
        x = self.conv_series(x)
        return x
        

class U_net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input shape  - ( 3, 256, 256)
        # output shape - (30, 256, 256)
        
        self.kernels = [3,3]
        self.number_of_filters_input = 3
        self.number_of_output_class = 30
        self.number_of_filters_level_0 = 64 
        self.number_of_filters_level_1 = 128
        self.number_of_filters_level_2 = 256
        self.number_of_filters_level_3 = 512
        self.number_of_filters_level_4 = 1024
        self.number_of_filters_level_5 = 2048
        
        # Level 0
        self.level_0 = Conv_series(self.number_of_filters_input, self.number_of_filters_level_0, kernels=self.kernels)
        
        # Level 0 -> Level 1
        self.level_0_to_1 = Down_step(self.number_of_filters_level_0, self.number_of_filters_level_1, kernels=self.kernels)

        # Level 1 -> Level 2
        self.level_1_to_2 = Down_step(self.number_of_filters_level_1, self.number_of_filters_level_2)
        
        # Level 2 -> Level 3
        self.level_2_to_3 = Down_step(self.number_of_filters_level_2, self.number_of_filters_level_3)
        
        # Level 3 -> Level 4
        self.level_3_to_4 = Down_step(self.number_of_filters_level_3, self.number_of_filters_level_4)
        
        # Level 4 -> Level 5
        self.level_4_to_5 = Down_step(self.number_of_filters_level_4, self.number_of_filters_level_5, dropout_prob=0.2)
        
        # Level 5 -> Level 4
        self.level_5_to_4 = Up_step(self.number_of_filters_level_5, self.number_of_filters_level_4)
        
        # Level 4 -> Level 3
        self.level_4_to_3 = Up_step(self.number_of_filters_level_4, self.number_of_filters_level_3)
        
        # Level 3 -> Level 2
        self.level_3_to_2 = Up_step(self.number_of_filters_level_3, self.number_of_filters_level_2)
        
        # Level 2 -> Level 1
        self.level_2_to_1 = Up_step(self.number_of_filters_level_2, self.number_of_filters_level_1)
        
        # Level 1 -> Level 0
        self.level_1_to_0 = Up_step(self.number_of_filters_level_1, self.number_of_filters_level_0, kernels=self.kernels)
        
        # Output
        self.output = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_output_class, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
    def forward(self, x):
        x_level_0_left = self.level_0(x)
        x_level_1_left = self.level_0_to_1(x_level_0_left)
        x_level_2_left = self.level_1_to_2(x_level_1_left)
        x_level_3_left = self.level_2_to_3(x_level_2_left)
        x_level_4_left = self.level_3_to_4(x_level_3_left)
        x_level_5_left = self.level_4_to_5(x_level_4_left)
        
        x_level_5_rght = x_level_5_left
        
        x_level_4_rght = self.level_5_to_4(x_level_5_rght,
                                           x_level_4_left)
        x_level_3_rght = self.level_4_to_3(x_level_4_rght,
                                           x_level_3_left)
        x_level_2_rght = self.level_3_to_2(x_level_3_rght,
                                           x_level_2_left)
        x_level_1_rght = self.level_2_to_1(x_level_2_rght,
                                           x_level_1_left)
        x_level_0_rght = self.level_1_to_0(x_level_1_rght,
                                           x_level_0_left)
        return self.output(x_level_0_rght)

def save_network_checkpoint(config, network, total_loss, accuracy, epoch):
    if not os.path.exists(config["CHECKPOINTS"]):
        os.mkdir(config["CHECKPOINTS"])
    if not os.path.exists(config["LOGS"]):
        os.mkdir(config["LOGS"])
        os.mkdir(os.path.join(config["LOGS"], "training"))
        os.mkdir(os.path.join(config["LOGS"], "validation"))
    
    now = str(datetime.datetime.now())
    
    torch.save(network, "checkpoints/" + now + "_epoch_" + str(epoch))
    
    out = open("logs/training/" + now + "_loss", "w")
    out.write(str(total_loss))
    out.close()
    
    out = open("logs/validation/" + now + "_error", "w")
    out.write(str(accuracy))
    out.close()
    
def epoch_training(network, optimizer, criterion, train_set_loader):
    network.train()
    total_loss = 0
    losses = []
    for data in train_set_loader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels.long())
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    network.eval()
    return sum(losses)/len(losses)
    
def training(config, network, train_data_loader, validation_data_loader, number_of_epochs=100):
    writer = SummaryWriter()
    #writer.add_graph(network)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(number_of_epochs):
        print("Epoch " + str(epoch))
        mean_loss = epoch_training(network, optimizer, criterion, train_data_loader)
        acc = accuracy(writer, network, validation_data_loader)
        print('Epoch ' + str(epoch) + ': Mean loss:' + str(mean_loss))
        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*acc) + '%')
        writer.add_scalar("Mean loss", mean_loss)
        save_network_checkpoint(config, network, mean_loss, acc, epoch)
    print('Finished Training')
    writer.close()

def accuracy(writer, network, validation_set_loader):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for data in validation_set_loader:
            validation_input, validation_output = data
            validation_input = validation_input.cuda()
            validation_output = validation_output.cuda()

            outputs = network(validation_input)
            outputs_flipped = network(torch.flip(validation_input, [3]))
            summed_outputs = outputs + torch.flip(outputs_flipped,[3])
            predicted = summed_outputs.argmax(dim=1)
            total += np.prod(validation_output.shape)
            correct += (predicted == validation_output.long()).sum().item()
    writer.add_image("Input", validation_input[0].cpu().view(3,256,256))
    writer.add_image("Output", ids_to_rgb_image_pytorch(validation_output[0].cpu(),id_to_colour_dict).view(3,256,256))
    writer.add_scalar("Accuracy", correct/total)
    return correct/total