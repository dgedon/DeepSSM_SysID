import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import time


def run_train(modelstate, loader_train, loader_valid, options, dataframe, path_general, file_name_general):
    def validate(loader):
        modelstate.model.eval()
        total_vloss = 0
        total_batches = 0
        total_points = 0
        with torch.no_grad():
            for i, (u, y) in enumerate(loader):
                u = u.to(options['device'])
                y = y.to(options['device'])
                vloss_ = modelstate.model(u, y)

                total_batches += u.size()[0]
                total_points += np.prod(u.shape)
                total_vloss += vloss_.item()

        return total_vloss / total_points  # total_batches

    def train(epoch):
        # model in training mode
        modelstate.model.train()
        # initialization
        total_loss = 0
        total_batches = 0
        total_points = 0

        for i, (u, y) in enumerate(loader_train):
            u = u.to(options['device'])
            y = y.to(options['device'])

            # set the optimizer
            modelstate.optimizer.zero_grad()
            # forward pass over model
            loss_ = modelstate.model(u, y)
            # NN optimization
            loss_.backward()
            modelstate.optimizer.step()

            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_loss += loss_.item()

            # output to console
            if i % train_options.print_every == 0:
                print(
                    'Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}'.format(
                        epoch, train_options.n_epochs, (i + 1), len(loader_train),
                        100. * (i + 1) / len(loader_train), lr, total_loss / total_points))  # total_batches

        return total_loss / total_points

    try:
        model_options = options['model_options']
        train_options = options['train_options']

        modelstate.model.train()
        # Train
        vloss = validate(loader_valid)
        all_losses = []
        all_vlosses = []
        best_vloss = vloss
        start_time = time.time()

        # Extract initial learning rate
        lr = train_options.init_lr

        # output parameter
        best_epoch = 0

        for epoch in range(0, train_options.n_epochs + 1):
            # Train and validate
            train(epoch)  # model, train_options, loader_train, optimizer, epoch, lr)
            # validate every n epochs
            if epoch % train_options.test_every == 0:
                vloss = validate(loader_valid)
                loss = validate(loader_train)
                # Save losses
                all_losses += [loss]
                all_vlosses += [vloss]

                if vloss < best_vloss:  # epoch == train_options.n_epochs:  #
                    best_vloss = vloss
                    # save model
                    path = path_general + 'model/'
                    file_name = file_name_general + '_bestModel.ckpt'
                    modelstate.save_model(epoch, vloss, time.clock() - start_time, path, file_name)
                    # torch.save(model.state_dict(), path + file_name)
                    best_epoch = epoch

                # Print validation results
                print('Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}'
                      '\tVal Loss: {:.3f}'.format(epoch, train_options.n_epochs, len(loader_train), len(loader_train),
                                                  100., lr, loss, vloss))

                # lr scheduler
                if epoch >= train_options.lr_scheduler_nstart:
                    if len(all_vlosses) > train_options.lr_scheduler_nepochs and \
                            vloss >= max(all_vlosses[int(-train_options.lr_scheduler_nepochs - 1):-1]):
                        # reduce learning rate
                        lr = lr / train_options.lr_scheduler_factor
                        # adapt new learning rate in the optimizer
                        for param_group in modelstate.optimizer.param_groups:
                            param_group['lr'] = lr
                        print('\nLearning rate adapted! New learning rate {:.3e}\n'.format(lr))
                # Early stoping condition
                if lr < train_options.min_lr:
                    break

    except KeyboardInterrupt:
        print('\n')
        print('-' * 89)
        print('Exiting from training early')
        # modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'interrupted_model.pt')
        print('-' * 89)

    # print best saved epoch model
    # print('\nBest model from epoch {} saved.'.format(best_epoch))

    # print time of learning
    time_el = time.time() - start_time
    # print('\nTotal learning time: {:2.0f}:{:2.0f} [min:sec]'.format(time_el // 60, time_el - 60 * (time_el // 60)))

    # save data in dictionary
    train_dict = {'all_losses': all_losses,
                  'all_vlosses': all_vlosses,
                  'best_epoch': best_epoch,
                  'total_epoch': epoch,
                  'train_time': time_el}
    # overall options
    dataframe.update(train_dict)

    return dataframe
