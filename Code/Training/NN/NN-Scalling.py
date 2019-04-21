# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2019-04-21 12:19:14

import torch

import torch
from torch.autograd import Variable


# class LinearRegressionModel(torch.nn.Module):

#     def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.layer1 = torch.nn.Linear(input_dim, n_hidden_1)
#         self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
#         self.layer3 = torch.nn.Linear(n_hidden_2, output_dim)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(
            input_dim, n_hidden_1), torch.nn.LeakyReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(
            n_hidden_1, n_hidden_2), torch.nn.LeakyReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_2, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# our model
model = LinearRegressionModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):

    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))

new_var = Variable(torch.Tensor([[4.0]]))
pred_y = model(new_var)
print("predict (after training)", 4, model(new_var).data[0][0])
