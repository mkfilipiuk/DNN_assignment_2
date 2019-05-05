import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import datetime
import os

from src.utils import flatten

class Conv_series(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3]):
        super().__init__()
        
        first_layer = [nn.Conv2d(number_of_filters_input, number_of_filters_output, kernels[0], padding=kernels[0] // 2, bias=False)]
        middle_layers =  flatten([[nn.BatchNorm2d(number_of_filters_output), nn.ReLU(inplace=True), nn.Conv2d(number_of_filters_output, number_of_filters_output, k, padding=k // 2, bias=False), nn.ReLU(inplace=True)] for k in kernels[1:]])
        end_layers = [nn.BatchNorm2d(number_of_filters_output), nn.ReLU(inplace=True)]
        
        self.bottom = nn.Sequential(
            *(first_layer + middle_layers + end_layers)
        )
        
    def forward(self, x):
        x = self.bottom(x)
        return x

class Down_step(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3]):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_series(number_of_filters_input, number_of_filters_output, kernels)
        )
        
    def forward(self, x):
        x = self.down(x)
        return x
    
class Up_step(nn.Module):
    def __init__(self, number_of_filters_input, number_of_filters_output, kernels = [3,3]):
        super().__init__()
        self.up = nn.ConvTranspose2d(number_of_filters_input, number_of_filters_output, 2, stride = 2)
        self.conv_series = Conv_series(number_of_filters_input, number_of_filters_output, kernels)
        
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
        
        self.number_of_filters_input = 3
        self.number_of_output_class = 30
        self.number_of_filters_level_0 = 128 
        self.number_of_filters_level_1 = 256
        #self.number_of_filters_level_2 = 512
        #self.number_of_filters_level_3 = 1024
        
        # Level 0
        self.level_0 = Conv_series(self.number_of_filters_input, self.number_of_filters_level_0)
        
        # Level 0 -> Level 1
        self.level_0_to_1 = Down_step(self.number_of_filters_level_0, self.number_of_filters_level_1)

        # Level 1 -> Level 2
        #self.level_1_to_2 = Down_step(self.number_of_filters_level_1, self.number_of_filters_level_2)
        
        # Level 2 -> Level 3
        #self.level_2_to_3 = Down_step(self.number_of_filters_level_2, self.number_of_filters_level_3)
        
        # Level 3 -> Level 2
        #self.level_3_to_2 = Up_step(self.number_of_filters_level_3, self.number_of_filters_level_2)
        
        # Level 2 -> Level 1
        #self.level_2_to_1 = Up_step(self.number_of_filters_level_2, self.number_of_filters_level_1)
        
        # Level 1 -> Level 0
        self.level_1_to_0 = Up_step(self.number_of_filters_level_1, self.number_of_filters_level_0)
        
        # Output
        self.last_BN = nn.BatchNorm2d(self.number_of_filters_level_0)
        self.output = nn.Conv2d(self.number_of_filters_level_0, 
                                        self.number_of_output_class, 
                                        3, 
                                        padding = 1, 
                                        bias=True)
        
    def forward(self, x):
        x_level_0_left = self.level_0(x)
        x = self.level_0_to_1(x_level_0_left)
        #x_level_2_left = self.level_1_to_2(x_level_1_left)
        #x = self.level_2_to_3(x_level_2_left)
        #x = self.level_3_to_2(x, x_level_2_left)
        #x = self.level_2_to_1(x, x_level_1_left)
        x = self.level_1_to_0(x, x_level_0_left)
        x = self.last_BN(x)
        x = self.output(x)
        return x

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
    for i, data in enumerate(train_set_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels.long())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    network.eval()
    return total_loss
    
def training(config, network, train, validation, number_of_epochs=100):
    train_input, train_output = train
    (validation_input, validation_input_flipped), (validation_output, validation_output_flipped) = validation
    train_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(train_input).transpose(1,3).contiguous(), torch.Tensor(train_output)), batch_size=32, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(validation_input).transpose(1,3).contiguous(), torch.Tensor(validation_output), torch.Tensor(validation_input_flipped).transpose(1,3).contiguous(), torch.Tensor(validation_output_flipped)), batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction = "sum")
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(number_of_epochs):
        print("Epoch " + str(epoch))
        total_loss = epoch_training(network, optimizer, criterion, train_data_loader)
        acc = accuracy(network, validation_data_loader)
        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*acc) + '%')
        save_network_checkpoint(config, network, total_loss, acc, epoch)
    print('Finished Training')

def accuracy(network, validation_set_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_set_loader:
            validation_input, validation_output, validation_input_flipped, validation_output_flipped = data
            validation_input = validation_input.cuda()
            validation_output = validation_output.cuda()
            validation_input_flipped = validation_input_flipped.cuda()

            outputs = network(validation_input)
            outputs_flipped = network(validation_input_flipped)
            _, predicted = torch.max((outputs.data+torch.flip(outputs_flipped, [2]))/2, 1)
            total += validation_input.view(-1).size(0)
            correct += (predicted == validation_output.long()).sum().item()
    return correct/total