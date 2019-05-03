import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        
        # input shape - (3,256,256)
        # output shape - (30, 256, 256)
        
        self.number_of_filters_input = 3
        self.number_of_output_class = 30
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
        self.up_conv_level_4_to_level_3 = nn.ConvTranspose2d(self.number_of_filters_level_4,
                                                             self.number_of_filters_level_3,
                                                             2,
                                                             stride = 2)
        
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
        
        self.up_conv_level_3_to_level_2 = nn.ConvTranspose2d(self.number_of_filters_level_3,
                                                             self.number_of_filters_level_2,
                                                             2,
                                                             stride = 2)
        
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
            
        self.up_conv_level_2_to_level_1 = nn.ConvTranspose2d(self.number_of_filters_level_2,
                                                             self.number_of_filters_level_1,
                                                             2,
                                                             stride = 2)
        
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
        
        self.up_conv_level_1_to_level_0 = nn.ConvTranspose2d(self.number_of_filters_level_1,
                                                             self.number_of_filters_level_0,
                                                             2,
                                                             stride = 2)
        
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
        x_level_3_right = F.relu(self.up_conv_level_4_to_level_3(x_level_4))
        
        # level 3
        x_level_3_right = torch.cat((x_level_3_left, x_level_3_right), 1)
        x_level_3_right = F.relu(self.level_3_conv_2(x_level_3_right))
        x_level_3_right = F.relu(self.level_3_conv_3(x_level_3_right))
        x_level_2_right = F.relu(self.up_conv_level_3_to_level_2(x_level_3_right))
        
        # level 2
        x_level_2_right = torch.cat((x_level_2_left, x_level_2_right), 1)
        x_level_2_right = F.relu(self.level_2_conv_2(x_level_2_right))
        x_level_2_right = F.relu(self.level_2_conv_3(x_level_2_right))
        x_level_1_right = F.relu(self.up_conv_level_2_to_level_1(x_level_2_right))
        
        # level 1
        x_level_1_right = torch.cat((x_level_1_left, x_level_1_right), 1)
        x_level_1_right = F.relu(self.level_1_conv_2(x_level_1_right))
        x_level_1_right = F.relu(self.level_1_conv_3(x_level_1_right))
        x_level_0_right = F.relu(self.up_conv_level_1_to_level_0(x_level_1_right))
        
        # level 0
        x_level_0_right = torch.cat((x_level_0_left, x_level_0_right), 1)
        x_level_0_right = F.relu(self.level_0_conv_2(x_level_0_right))
        x_level_0_right = F.relu(self.level_0_conv_3(x_level_0_right))
        x_level_0_right = F.relu(self.level_0_conv_4(x_level_0_right))

        return x_level_0_right

def epoch_training(network, optimizer, criterion, train_set_loader):
    network.train()
    for i, data in enumerate(train_set_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
    network.eval()
    
def training(network, train, validation, number_of_epochs=100):
    train_input, train_output = train
    (validation_input, validation_input_flipped), (validation_output, validation_output_flipped) = validation
    train_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(train_input).transpose(1,3).contiguous(), torch.Tensor(train_output)), batch_size=5, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(validation_input).transpose(1,3).contiguous(), torch.Tensor(validation_output), torch.Tensor(validation_input_flipped).transpose(1,3).contiguous(), torch.Tensor(validation_output_flipped)), batch_size=50, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    print('Epoch ' + str(1) + ': Accuracy of the network:' + str(100*accuracy(network, validation_data_loader)) + '%')
    for epoch in range(number_of_epochs):
        print("Epoch " + str(epoch))
        epoch_training(network, optimizer, criterion, train_data_loader)
        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*accuracy(network, validation_data_loader)) + '%')
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
            _, predicted = torch.max(outputs.data+torch.flip(outputs_flipped, [2])/2, 1)
            total += validation_input.view(-1).size(0)
            correct += (predicted == validation_output.long()).sum().item()
    return correct/total