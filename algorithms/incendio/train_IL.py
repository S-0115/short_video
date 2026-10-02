import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from Incendio_model import BA_Actor, BM_Actor, Critic,Trainer_IL

from utils import custom_collate_fn_BA, custom_collate_fn_BM
from exp_dataset import ExperienceDataset
from run_incendio import test_user_samples
import multiprocess as mp

learning_rate = 1e-4
weight_decay = 1e-5
grad_accum_steps = 5
report_loss_per_steps = 100

model_save_epoch = 20

from config_algorithm import beta
nn_model_save_path= './model_' + str(beta) + '/IL/'

exp_pool_path = './exp_pool_' + str(beta) + '/'

# NN_MODEL_BM = nn_model_save_path + 'bm_agent/bm_actor_epoch_' + '250' + '.pth'
# NN_MODEL_BA = nn_model_save_path + 'ba_agent/ba_actor_epoch_' + '250' + '.pth'

start_epoch = 0

NN_MODEL_BM = None
NN_MODEL_BA = None

USE_GPU = torch.cuda.is_available()

def test_model(epoch):
    with torch.no_grad():
        dataset_dir = '../data/dataset_2s_test'
        chunklength = 2000.

        os.system(f'python run_incendio.py '
                  f'--trace sampled_4G '
                  f'--dataset_dir {dataset_dir} '
                  f'--epoch {epoch} '
                  f'--chunklength {chunklength} '
                  f'--train_type IL')

def run_with_timeout(func, timeout, epoch):
    p = mp.Process(target=func, args=(epoch,))
    p.start()
    p.join()

    # if p.is_alive():
    #     p.terminate()
    #     p.join()
    #     print('========= time out ===========')

def adapt(num_epoch):
    print('begin IL training')
    # =============== BM-AGENT =================
    bm_actor = BM_Actor()

    # 创建自定义数据集
    data_bm, labels_bm = load_experience(exp_pool_path + 'exp_pool_BM.txt')
    custom_dataset_bm = ExperienceDataset(data_bm, labels_bm)
    print('BM EXP DATASET LOADED')
    trainer_bm = Trainer_IL(learning_rate, weight_decay, bm_actor)

    if USE_GPU:
        trainer_bm.cuda()
        trainer_bm.loss_function = trainer_bm.loss_function.cuda()
    # 创建 DataLoader

    dataloader_bm = DataLoader(custom_dataset_bm, batch_size=512, shuffle=True, collate_fn=custom_collate_fn_BM, drop_last=False)
    print('BM DATALOADER INITIALIZED')
    if NN_MODEL_BM is not None:
        trainer_bm.load_model(NN_MODEL_BM)
        print('model restore')

    # =============== BA-AGENT =================
    ba_actor = BA_Actor()

    # 创建自定义数据集
    data_ba, labels_ba = load_experience(exp_pool_path + 'exp_pool_BA.txt')
    custom_dataset_ba = ExperienceDataset(data_ba, labels_ba)
    print('BA EXP DATASET LOADED')
    trainer_ba = Trainer_IL(learning_rate, weight_decay, ba_actor)

    if USE_GPU:
        trainer_ba.cuda()
        trainer_ba.loss_function = trainer_ba.loss_function.cuda()
    # 创建 DataLoader

    dataloader_ba = DataLoader(custom_dataset_ba, batch_size=512, shuffle=True, collate_fn=custom_collate_fn_BA, drop_last=False)
    print('BA DATALOADER INITIALIZED')
    if NN_MODEL_BA is not None:
        trainer_ba.load_model(NN_MODEL_BA)
        print('model restore')

    for epoch in range(start_epoch, num_epoch):
        print('=' * 10, 'epoch:', epoch, '=' * 10)
        print('-' * 10, 'training BM-AGENT', '-' * 10)
        train_losses_bm = []
        for step, batch in enumerate(dataloader_bm):
            states, bts, labels = process_batch(batch)
            # print(states)
            states = states.reshape(states.shape[0], 1, 4, 5)
            # print(states)

            if USE_GPU:
                states, bts, labels = states.cuda(), bts.cuda(), labels.cuda()
            # print(states.shape, labels.shape, bts.shape) #torch.Size([2048, 1, 4, 5]) torch.Size([2048]) torch.Size([2048, 5, 1])

            loss = trainer_bm.forward(states, bts, labels)
            train_losses_bm.append(loss.item())
            # print(train_losses_bm[-5:])

            train_loss = loss
            train_loss.backward()

            trainer_bm.optimizer.step()
            trainer_bm.optimizer.zero_grad()

            if step % report_loss_per_steps == 0:
                mean_train_loss = np.mean(train_losses_bm)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        print('-' * 10, 'training BA-AGENT', '-' * 10)
        train_losses_ba = []
        for step, batch in enumerate(dataloader_ba):
            states, bts, labels = process_batch(batch)

            if USE_GPU:
                states, bts, labels = states.cuda(), bts.cuda(), labels.cuda()
            # print(states.shape, labels.shape, bts.shape) # torch.Size([512, 7, 5]) torch.Size([512]) torch.Size([512, 5, 1])

            loss = trainer_ba.forward(states, bts, labels)
            train_losses_ba.append(loss.item())

            train_loss = loss
            train_loss.backward()

            trainer_ba.optimizer.step()
            trainer_ba.optimizer.zero_grad()

            if step % report_loss_per_steps == 0:
                mean_train_loss = np.mean(train_losses_ba)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        if epoch % model_save_epoch == 0 or epoch == num_epoch - 1:
            print('-' * 10, 'testing', '-' * 10)
            BM_MODEL_SAVE_PATH = nn_model_save_path + 'bm_agent/bm_actor_epoch_' + str(epoch) + '.pth'
            trainer_bm.save_model(BM_MODEL_SAVE_PATH)

            BA_MODEL_SAVE_PATH = nn_model_save_path + 'ba_agent/ba_actor_epoch_' + str(epoch) + '.pth'
            trainer_ba.save_model(BA_MODEL_SAVE_PATH)

            run_with_timeout(test_model, 600, epoch)

def load_experience(exp_pool_file_path):
    datas = []
    labels = []
    with open(exp_pool_file_path, 'r') as f:
        for line in f:
            list_exp = json.loads(line.strip())
            datas.append(list_exp[:-1])
            labels.append(list_exp[-1])
    f.close()

    return datas, labels
def process_batch(batch):
    states = batch[0]
    labels = batch[1]

    bt = []
    for input in states:
        bt.append(list(input[0]))
    bt = torch.tensor(bt).reshape(len(bt), 5, 1)

    return states, bt, labels

def main():
    adapt(101)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()