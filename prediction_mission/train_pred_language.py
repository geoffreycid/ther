import os
import collections
import random
import operator

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import tensorboardX as tb
from sklearn.metrics import f1_score

import sys
sys.path.append('../')
import models.predmissionnet as prednet


def optimization(train_memory, test_memory, test_memory_with_hold_out_missions, config):

    # IMC
    imc = collections.namedtuple("ImMis", ["state", "mission", "target"])

    # Config
    n_iterations = config["n_iterations"]
    batch_size = config["batch_size"]

    len_train = len(train_memory)

    # Choose the device
    if "device" in config:
        device = config["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Early stopping parameters
    earlystopping = config["earlystopping"]
    iterations_before_early_stopping = config["iterations_before_earlystopping"]

    # word indices
    word2idx = config["word2idx"]
    idx2word = {}
    for key, ind in config["word2idx"].items():
        idx2word[ind] = key

    net = prednet.PredMissionRNN(c=4, frames=config["frames"], n_words=config["n_words"],
                                 word_embedding_size=config["word_embedding_size"],
                                 hidden_size=config["hidden_size"],
                                 teacher_forcing_ratio=config["teacher_forcing_ratio"],
                                 word2idx=word2idx,
                                 idx2word=idx2word,
                                 lr=config["lr"],
                                 weight_decay=config["weight_decay"]).to(device)

    # Directories for the logs and save
    dir_logs = config["dir_logs"] + "/len_train_{}".format(len_train)
    dir_save_model = config["dir"] + "/saved_model/" + "len_train_{}".format(len_train)

    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    # Tensorboard
    writer = tb.SummaryWriter(dir_logs)

    # Log
    with open(dir_logs + "/sentences.csv", "a") as log:
        log.write("{},{},{}\n".format("index", "prediction", "truth"))

    # Optimization steps
    steps_done = 0

    for _ in range(int(n_iterations)):
        imcs = random.sample(train_memory, batch_size)

        batch_imcs = imc(*zip(*imcs))
        batch_states = torch.cat(batch_imcs.state)[:, 12:].to(device)

        # Add the END token before padding
        targets = []
        for tar in batch_imcs.target:
            targets.append(torch.cat((tar, word2idx["END"] * torch.ones(1, dtype=torch.long))))
        # dim = (seq_len, batch)
        batch_targets = nn.utils.rnn.pad_sequence(targets).to(device)

        net.optimizer.zero_grad()

        loss = net.forward(batch_states, batch_targets)
        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in net.parameters():
            param.grad.data.clamp_(-10, 10)
        net.optimizer.step()

        steps_done += 1

        if steps_done % config["log_every"] == 0:

            writer.add_scalar("Loss", loss.item(), global_step=steps_done)
            # Accuracy on train missions but different images
            imcs = test_memory
            batch_imcs = imc(*zip(*imcs))
            batch_states = torch.cat(batch_imcs.state)[:, 12:].to(device)
            pred_idxs = net.prediction_mission(batch_states)

            acc_whole_mission_train_mission = 0
            accuracy_train_mission = 0
            acc_per_attrib = np.zeros(4)
            for ind, mission in enumerate(batch_imcs.target):
                count = 0
                for i, idx in enumerate(torch.sort(mission[-4:])[0]):
                    # first idx => color then type then shade then size
                    match = idx in pred_idxs[ind]
                    accuracy_train_mission += match
                    count += match
                    acc_per_attrib[i] += match
                if count == 4:
                    acc_whole_mission_train_mission += 1
            accuracy_train_mission /= len(batch_imcs.target) * 4
            acc_whole_mission_train_mission /= len(batch_imcs.target)
            acc_per_attrib /= len(batch_imcs.target)

            writer.add_scalar("acc train missions", accuracy_train_mission, global_step=steps_done)
            writer.add_scalar("whole mission acc train missions", acc_whole_mission_train_mission, global_step=steps_done)
            writer.add_scalar("color acc train missions", acc_per_attrib[0], global_step=steps_done)
            writer.add_scalar("type acc train missions", acc_per_attrib[1], global_step=steps_done)
            writer.add_scalar("shade acc train missions", acc_per_attrib[2], global_step=steps_done)
            writer.add_scalar("size acc train missions", acc_per_attrib[3], global_step=steps_done)

            # Accuracy on hold out missions
            imcs = test_memory_with_hold_out_missions
            batch_imcs = imc(*zip(*imcs))
            batch_states = torch.cat(batch_imcs.state)[:, 12:].to(device)
            pred_idxs = net.prediction_mission(batch_states)

            acc_whole_mission_test_mission = 0
            accuracy_test_mission = 0
            acc_per_attrib = np.zeros(4)
            for ind, mission in enumerate(batch_imcs.target):
                count = 0
                for i, idx in enumerate(torch.sort(mission[-4:])[0]):
                    # first idx => color then type then shade then size
                    match = idx in pred_idxs[ind]
                    accuracy_test_mission += match
                    count += match
                    acc_per_attrib[i] += match
                if count == 4:
                    acc_whole_mission_test_mission += 1
            accuracy_test_mission /= len(batch_imcs.target) * 4
            acc_whole_mission_test_mission /= len(batch_imcs.target)
            acc_per_attrib /= len(batch_imcs.target)

            writer.add_scalar("acc hold out missions", accuracy_test_mission, global_step=steps_done)
            writer.add_scalar("whole mission acc hold out missions", acc_whole_mission_test_mission, global_step=steps_done)
            writer.add_scalar("color acc hold out missions", acc_per_attrib[0], global_step=steps_done)
            writer.add_scalar("type acc hold out missions", acc_per_attrib[1], global_step=steps_done)
            writer.add_scalar("shade acc hold out missions", acc_per_attrib[2], global_step=steps_done)
            writer.add_scalar("size acc hold out missions", acc_per_attrib[3], global_step=steps_done)

            #writer.add_scalars('acc_whole_mission', {'train_mission': acc_whole_mission_train_mission,
            #                                         'holdout_mission': acc_whole_mission_test_mission},
            #                   global_step=steps_done)

            #writer.add_scalars('accuracy', {'train_mission': accuracy_train_mission,
            #                                         'holdout_mission': accuracy_test_mission},
            #                   global_step=steps_done)

            pred_sentences = []
            for i in range(len(pred_idxs)):
                idxs = pred_idxs[i]
                sentence = ''
                for idx in idxs:
                    sentence += ' ' + idx2word[idx.item()]
                pred_sentences.append(sentence[1:])

            true_idxs = nn.utils.rnn.pad_sequence(batch_imcs.target).to(device).t()
            true_sentences = []
            for i in range(len(true_idxs)):
                idxs = true_idxs[i]
                sentence = ''
                for idx in idxs:
                    if idx == word2idx["PAD"]:
                        break
                    sentence += ' ' + idx2word[idx.item()]
                true_sentences.append(sentence[1:])

            # Log on a txt file
            with open(dir_logs + "/sentences.csv".format(len_train), "a") as log:
                for i in range(len(pred_sentences)):
                    log.write("{},{},{}\n".format(i, pred_sentences[i], true_sentences[i]))

        # Save the weights
        if steps_done % config["save_every"] == 0:
            path_to_save = os.path.join(dir_save_model, "{}_steps_{}_acc.pt".format(steps_done,
                                                                                    acc_whole_mission_test_mission))
            torch.save(net.state_dict(), path_to_save)

        if steps_done > n_iterations:
            break

    writer.close()
