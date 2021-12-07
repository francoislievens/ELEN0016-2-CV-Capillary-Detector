import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from UNet_DataHandler import DataBuilder
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from UNet import UNet
import sys
import matplotlib.pyplot as plt
from utils import avg_smoothing, count_peaks_2d
from UNet import load_prev_unet
import cv2



def train(data_path='UNet_Dataset',
          model_path='Model',
          name='UNet_A',
          batch_size=15,
          learning_rate=1e-6,
          weight_decay=0,
          num_workers=4,
          target_epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model, loss and optimizer:
    model = UNet(input_filters=1).to(device)
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # Data Loader
    train_set = DataBuilder(dataset_path=data_path,
                            train_prop=0.8,
                            train=True,
                            data_augmentation=True)
    test_set = DataBuilder(dataset_path=data_path,
                           train_prop=0.8,
                           train=False,
                           data_augmentation=False)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )


    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int((batch_size / 0.8) * 0.2),
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    test_iterator = iter(test_loader)

    model, optimizer, starting_epoch = load_prev_unet(model, optimizer, name, model_path, device)

    # Main loop
    test_batch_idx = 0
    for e in range(starting_epoch, target_epoch):
        print('Training at epoch {} / {}\n'.format(e + 1, target_epoch))
        loop = tqdm(train_loader, leave=True)

        train_loss = []
        test_loss = []
        for batch_id, (img, label) in enumerate(loop):

            # Training part
            model.train()
            img = img.to(device)
            label = label.to(device)
            preds = model(img).reshape((-1, 240, 240))
            loss = loss_fn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # Testing part
            if test_batch_idx >= len(test_iterator):
                test_iterator = iter(test_loader)
                test_batch_idx = 0
            test_batch_idx += 1
            model.eval()
            with torch.no_grad():
                img, label = next(test_iterator)
                img = img.to(device)
                label = label.to(device)
                preds = model(img).reshape((-1, 240, 240))
                loss = loss_fn(preds, label)
                test_loss.append(loss.item())
            loop.set_postfix(loss=train_loss[-1])

        # Save logs
        f = open('{}/{}/logs.csv'.format(model_path, name), 'a')
        for i in range(len(train_loss)):
            f.write('{},{},{}\n'.format(e, train_loss[i], test_loss[i]))
        f.close()
        # Save model
        torch.save(model.state_dict(), '{}/{}/model_weights.pt'.format(model_path, name))
        torch.save(optimizer.state_dict(), '{}/{}/optimizer_weights.pt'.format(model_path, name))


def plot_UNet_logs(model_path='model',
                   model_name='UNet_A'):
    logs = pd.read_csv('{}/{}/logs.csv'.format(model_path, model_name), sep=',')

    # Get data
    x_axis = np.arange(logs.shape[0])
    train_loss = logs['train_loss'].to_numpy()
    test_loss = logs['test_loss'].to_numpy()

    # Apply smoothing
    window_size = 50
    train_loss = avg_smoothing(train_loss, window_size)[50:-50]
    test_loss = avg_smoothing(test_loss, window_size)[50:-50]

    # Get epoch changes index
    ep_idx = []
    val = 0
    for i in range(0, logs.shape[0]):
        if logs.iloc[i]['epoch'] > val:
            val = logs.iloc[i]['epoch']
            ep_idx.append(i)

    # Plot it
    plt.plot(x_axis[50:-50], train_loss, color='darkgreen', label='Train Loss', linewidth=0.5)
    plt.plot(x_axis[50:-50], test_loss, color='red', label='Test Loss', linewidth=0.5)
    for i in range(0, len(ep_idx)):
        if i == 0:
            plt.axvline(x=ep_idx[i], label='Epoch change', c='blue', linewidth=0.3)
        else:
            plt.axvline(x=ep_idx[i], c='blue', linewidth=0.3)
    plt.title('UNet training curves')
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    plt.yscale('log')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.savefig('Figs/training_curves_{}.png'.format(model_name))
    plt.show()
    plt.close()


def evaluate(data_path='UNet_Dataset',
             model_path='Model',
             name='UNet_A'):
    # Loading model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(input_filters=1).to(device)
    model.load_state_dict(torch.load('{}/{}/model_weights.pt'.format(model_path, name), map_location=device))

    # Loading the testing set
    dataset = DataBuilder(dataset_path=data_path,
                          train_prop=0.8,
                          train=False,
                          data_augmentation=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=True
    )

    model.eval()
    for batch_id, (img, label) in enumerate(data_loader):

        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(img)
            #pred = torch.where(pred > 0.0, pred, 0.0).to(torch.float)
            #pred[pred < 0.0] = 0.0
            pred_count = torch.sum(pred).cpu().numpy()
            label_count = torch.sum(label.reshape(pred.shape), dim=(1, 2)).cpu().numpy()

        img = img.reshape(240, 240).cpu().numpy()
        label = label.reshape(240, 240).cpu().numpy()
        pred = pred.reshape(240, 240).cpu().numpy()

        # Cell count by peaks finding
        nb_cell = count_peaks_2d(pred)
        nb_cell_target = count_peaks_2d(label)

        #print('pred_count: {}, label_count: {}'.format(np.sum(pred_count) / 100, np.sum(label_count)/100))
        print('pred_count: {}, label_count: {}'.format(nb_cell, nb_cell_target))
        cv2.imshow('img', img)
        cv2.imshow('label', label)
        cv2.imshow('pred', pred)
        cv2.waitKey()







if __name__ == '__main__':

    #train(target_epoch=30, name='UNet_B')
    #evaluate(name='UNet_A')
    plot_UNet_logs(model_name='UNet_D')
