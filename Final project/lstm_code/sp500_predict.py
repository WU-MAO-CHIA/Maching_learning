# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 05:13:33 2021

@author: narut
"""
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from future_dataset import FutureDataset
from future_lstm import FutureLSTM

def train(net, train_loader, optimizer, criterion, epoch_idx, device=None):
  net.train()
  running_loss = 0.0
  batch_cnt = 0
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    if device != None:
      inputs, labels = inputs.to(device), labels.to(device)
      
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(inputs.double())  # Forward
    loss = criterion(outputs, labels)
    loss.backward()   # Backprop
    optimizer.step()  # Update parameters

    running_loss += loss.item()
    batch_cnt = batch_idx

    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
        epoch_idx, batch_idx * len(inputs), len(train_loader.dataset),      
        100. * batch_idx / len(train_loader), loss.item()))
      
  return (running_loss / batch_cnt)

def validate(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  batch_cnt = 0
  pred = []
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs.double())
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred += outputs.tolist()   
      batch_cnt = batch_idx
      
  test_loss /= batch_cnt

  print('\nValid set: Average loss: {:.4f}\n'.format(test_loss)) 
  return test_loss

def test(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  batch_cnt = 0
  pred = []
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs.double())
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred += outputs.tolist()   
      batch_cnt = batch_idx
      
  test_loss /= batch_cnt

  print('\nValid set: Average loss: {:.4f}\n'.format(test_loss)) 
  return pred

def main():
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    
    # Read data from csv file
    df_all = pd.read_csv("./training_data.csv")
    
    # Select target columns
    """
    columns = df.columns.tolist()
    columns.remove('Date')
    columns.remove('SP500_Index')
    columns.remove('SP500_close_m')
    columns.remove('DJ_close')
    columns.remove('DJ_close_m')
    columns.remove('Nasdaq_close')
    columns.remove('Nasdaq_close_m')
    columns.remove('VIX_Close')
    """
    df_all['SP500_Index_Growth'] = df_all['SP500_Index_Growth']*100
    tar_columns = df_all[['SP500_Futures_Volume_std', 'SP500_Futures_ROC', 'SP500_Futures_close_std', 
                'NASDAQ_Futures_Volume_std', 'NASDAQ_Futures_ROC', 'NASDAQ_Futures_close_std',
                'SP500_Index_Growth']]
    data_df = tar_columns.copy()
    data = data_df.to_numpy()
    
    train_data = data[:3900, :]
    val_data = data[3900: , :]
    
    """
    idx = [i for i in range(4867)]
    val_idx = random.sample(range(4867), 980)
    train_idx = list(set(idx) - set(val_idx))

    train_data = data[train_idx]
    val_data = data[val_idx]
    """
    
    # Training config
    batch_size = 32
    epoch_num = 300
    lr = 1e-4
    
    # Datasets and dataloaders
    train_set = FutureDataset(train_data, 60)
    val_set = FutureDataset(val_data, 60)  
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    train_loader_val = DataLoader(train_set, batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = FutureLSTM(input_size=6, hidden_size=1024, device=device, num_layers=3).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    
    
    # Training
    loss_list = []
    train_err_list = []
    test_err_list = []
  
    for epoch in range(1, epoch_num + 1):
      loss = train(net, train_loader, optimizer, criterion, epoch, device)
      loss_list.append(loss)
      
      train_err = validate(net, train_loader_val, criterion, device)
      train_err_list.append(train_err)
      test_err = validate(net, val_loader, criterion, device)
      test_err_list.append(test_err)
      
      if epoch % 10 == 0:
        torch.save(net.state_dict(), 'future_lstm_{}.pt'.format(epoch))
    
  
    # Save parameters of the model
    torch.save(net.state_dict(), 'future_lstm_full.pt')
    #net.load_state_dict(torch.load('future_lstm_1.pt'))
    
    # Predict
    result = test(net, train_loader_val, criterion, device)
    print(result)
    print(len(result))
    fig , ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (10, 3.5)
    plt.plot(range(len(train_set)), train_set.y.tolist(), label = "Label", color = "orange", linewidth = 0.5)
    plt.plot(range(len(train_set)), result, label = "Predict", color = "blue", linewidth = 0.5)
    plt.title("Result of Prediction")
    #plt.ylabel("Error Rate(%)")
    #plt.xlabel("Epoch")
    leg = ax.legend(loc='lower right') 
    plt.savefig('predict_train.png')
    plt.show()

    result1 = test(net, val_loader, criterion, device)
    fig , ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (10, 3.5)
    plt.plot(range(len(val_set)), val_set.y.tolist(), label = "Label", color = "orange", linewidth = 0.5)
    plt.plot(range(len(val_set)), result1, label = "Predict", color = "blue", linewidth = 0.5)
    plt.title("Result of Prediction")
    #plt.ylabel("Error Rate(%)")
    #plt.xlabel("Epoch")
    leg = ax.legend(loc='lower right') 
    plt.savefig('predict.png')
    plt.show()
    
    
    # Plot Accuracy
    print("=== Show loss plot ===>>")
    fig , ax = plt.subplots()
    #plt.rcParams["figure.figsize"] = (8, 3.5)
    plt.plot(range(len(train_err_list)), train_err_list, label = "training loss", color = "blue", linewidth = 0.5)
    plt.plot(range(len(test_err_list)), test_err_list, label = "testing loss", color = "orange", linewidth = 0.5)
    plt.title("Loss of the Model")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    leg = ax.legend(loc='lower right') 
    plt.savefig('lstm_err_rate.png')
    plt.show()
    
    print(" ")
    
    # Plot learning curve
    print("=== Show learning plot ===>>")
    plt.plot(loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Learning Curve")
    plt.savefig('lstm_lc.png')
    plt.show()
    
    print(" ")
    
    

    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
  main()
  