import os
import collections
import operator
import numpy as np
import torch
import models.predmissionnet as prednet
import torch.nn.functional as F
import tensorboardX as tb


#config = {
#    "n_epochs": 1000,
#    "batch_size": 128,
#    "imc": 0,
#    "frames": 4,
#    "channels": 4,
#    "num_types": 2,
#    "num_colors": 6,
#    "num_seniority": 5,
#    "num_size": 5,
#    "tuple_imc": collections.namedtuple("imc", ["state", "mission", "target"]),
#    "dir": "/home/gcideron/home/gcideron/visual_her/logs_pred"
#}


def optimization(net, train_memory, test_memory, config):

    # Config
    n_iterations = config["n_iterations"]
    save_every = config["save_every"]
    batch_size = config["batch_size"]

    len_train = len(train_memory)
    len_test = len(test_memory)

    num_types = config["num_types"]
    num_colors = config["num_colors"]
    num_seniority = config["num_seniority"]
    num_size = config["num_size"]

    # Seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Early stopping parameters
    earlystopping = config["earlystopping"]
    iterations_before_early_stopping = config["iterations_before_earlystopping"]

    if config["imc"]:
        use_imc = 1
        use_onehot = 0
    else:
        use_onehot = 1
        use_imc = 0


    # Writer to tensorboard
    if use_imc:
        dir = config["dir"] + "/" + "imc" + "/" + "len_train_{}".format(len_train)
    else:
        dir = config["dir"] + "/" + "onehot" + "/" + "len_train_{}".format(len_train)

    if not os.path.exists(dir):
        os.makedirs(dir)
    writer = tb.SummaryWriter(dir)

    # Keep only the last frames
    keep = (config["frames"] - 1) * config["channels"]

    # Choose the right device
    if "device" in config:
        device = config["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    # List of all possible missions
    missions_type = F.one_hot(torch.arange(num_types))
    missions_color = F.one_hot(torch.arange(num_colors))
    missions_seniority = F.one_hot(torch.arange(num_seniority))
    missions_size = F.one_hot(torch.arange(num_size))

    all_possible_missions = []
    for i in range(missions_type.shape[0]):
        for j in range(missions_color.shape[0]):
            for k in range(missions_seniority.shape[0]):
                for l in range(missions_size.shape[0]):
                    all_possible_missions.append(
                        torch.cat((missions_type[i], missions_color[j], missions_seniority[k], missions_size[l])))
    all_possible_missions = torch.stack(all_possible_missions, dim=0).to(device).float()

    # Named tuple image mission correspondence (imc)
    tuple_imc = config["tuple_imc"]

    # Optimization steps
    steps_done = 0

    if use_imc:
        test_memory_with_good_mission = []
        for imc in test_memory:
            if imc.target == 1:
                test_memory_with_good_mission.append(imc)

        # Skew ratio to sample the same amount of 0 and 1
        train_targets = torch.cat(tuple_imc(*zip(*train_memory)).target).cpu().numpy()
        idx_targets_corres = np.argwhere(train_targets == 1).reshape(-1)
        idx_targets_no_corres = np.argwhere(train_targets == 0).reshape(-1)

    losses = []
    test_accs = np.array([])

    while True:
        beg_ind = 0
        end_ind = batch_size
        for i in range(len_train // batch_size):

            if use_imc:
                # Sample the same amount of 0 and 1
                batch_size_corres = batch_size // 2
                idx_corres = np.random.choice(idx_targets_corres, size=batch_size_corres)
                idx_no_corres = np.random.choice(idx_targets_no_corres, size=batch_size - batch_size_corres)
                op_corres = operator.itemgetter(*idx_corres)
                op_no_corres = operator.itemgetter(*idx_no_corres)
                imcs_corres = op_corres(train_memory)
                imcs_no_corres = op_no_corres(train_memory)
                imcs = imcs_corres + imcs_no_corres

            if use_onehot:
                imcs = train_memory[beg_ind:end_ind]

            batch_imcs = tuple_imc(*zip(*imcs))
            batch_states = torch.cat(batch_imcs.state)
            # Keep only the last frame
            batch_states = batch_states[:, keep:].to(device)
            # For IMC only for onehot do not use batch mission
            if use_imc:
                batch_missions = torch.cat(batch_imcs.mission).to(device)
                batch_targets = torch.cat(batch_imcs.target).to(device)

            elif use_onehot:
                batch_targets = torch.cat(batch_imcs.target).to(device)
                batch_type_targets = batch_targets[:, :num_types]
                batch_type_targets = batch_type_targets.argmax(1)

                batch_color_targets = batch_targets[:, num_types: num_types + num_colors]
                batch_color_targets = batch_color_targets.argmax(1)

                batch_seniority_targets = batch_targets[:,
                                          num_types + num_colors:num_types + num_colors + num_seniority]
                batch_seniority_targets = batch_seniority_targets.argmax(1)

                batch_size_targets = batch_targets[:, num_types + num_colors + num_seniority:]
                batch_size_targets = batch_size_targets.argmax(1)

            if use_imc:
                batch_predictions = net.image_mission_correspondence(batch_states, batch_missions)

            if use_onehot:
                batch_type_predictions, batch_color_predictions, batch_seniority_predictions, batch_size_predictions = net(
                    batch_states)

            if use_imc:
                loss = net.criterion(batch_predictions, batch_targets)
                net.optimizer_imc.zero_grad()
                loss.backward()
                net.optimizer_imc.step()

            elif use_onehot:
                net.optimizer.zero_grad()
                type_loss = net.criterion(batch_type_predictions, batch_type_targets)
                color_loss = net.criterion(batch_color_predictions, batch_color_targets)
                seniority_loss = net.criterion(batch_seniority_predictions, batch_seniority_targets)
                size_loss = net.criterion(batch_size_predictions, batch_size_targets)
                # loss = torch.sum([type_loss, color_loss, seniority_loss, size_loss])
                loss = sum([type_loss, color_loss, seniority_loss, size_loss])
                loss.backward()
                net.optimizer.step()

            beg_ind = end_ind
            end_ind += batch_size

            steps_done += 1

            if i % 100 == 0:
                losses.append(float(loss))

            if steps_done % save_every == 0:

                if use_imc:
                    batch_imcs = tuple_imc(*zip(*test_memory_with_good_mission))
                    batch_states = torch.cat(batch_imcs.state)[:, keep:].to(device)
                    batch_missions = torch.cat(batch_imcs.mission).to(device)
                    # Prediction of the missions
                    batch_pred_missions = net.find_best_mission(batch_states, all_possible_missions)
                    accuracy = torch.all(torch.eq(batch_pred_missions, batch_missions), dim=1)
                    accuracy = accuracy.float().mean().cpu().numpy()
                    writer.add_scalar("Accuracy", accuracy, global_step=steps_done)
                    test_accs = np.append(test_accs, accuracy)
                    # Early stopping
                    if test_accs.size > iterations_before_early_stopping \
                            and np.sum(test_accs[-earlystopping] < test_accs[-earlystopping:]) == 0:
                        print("Early stopping")
                        break

                    #print("Accuracy: {}".format(round(float(accuracy), 3)))

                elif use_onehot:
                    batch_imcs = tuple_imc(*zip(*test_memory))
                    batch_states = torch.cat(batch_imcs.state)
                    # Keep only the last frame
                    batch_states = batch_states[:, keep:].to(device)

                    # Predictions
                    batch_type_predictions, batch_color_predictions, batch_seniority_predictions, batch_size_predictions \
                        = net.prediction(batch_states)

                    batch_type_predictions_onehot = torch.eye(num_types)[batch_type_predictions]
                    batch_color_predictions_onehot = torch.eye(num_colors)[batch_color_predictions]
                    batch_seniority_predictions_onehot = torch.eye(num_seniority)[batch_seniority_predictions]
                    batch_size_predictions_onehot = torch.eye(num_size)[batch_size_predictions]
                    miss = (batch_type_predictions_onehot, batch_color_predictions_onehot, batch_seniority_predictions_onehot,
                            batch_size_predictions_onehot)
                    batch_mission_predictions = torch.cat(miss, dim=1).to(device)

                    # Targets
                    batch_targets = torch.cat(batch_imcs.target).to(device)
                    batch_type_targets = batch_targets[:, :num_types].max(1)[1]
                    batch_color_targets = batch_targets[:, num_types:num_types + num_colors].max(1)[1]
                    batch_seniority_targets = \
                    batch_targets[:, num_types + num_colors:num_types + num_colors + num_seniority].max(1)[1]
                    batch_size_targets = batch_targets[:, num_types + num_colors + num_seniority:].max(1)[1]

                    # Compute accuracies
                    acc_type = float(torch.eq(batch_type_predictions, batch_type_targets).sum()) / len_test
                    acc_color = float(torch.eq(batch_color_predictions, batch_color_targets).sum()) / len_test
                    acc_seniority = float(torch.eq(batch_seniority_predictions, batch_seniority_targets).sum()) / len_test
                    acc_size = float(torch.eq(batch_size_predictions, batch_size_targets).sum()) / len_test
                    acc_total = float(torch.all(torch.eq(batch_mission_predictions, batch_targets), dim=1).sum()) / len_test
                    writer.add_scalar("Accuracy", acc_total, global_step=steps_done)
                    writer.add_scalar("Accuracy type", acc_type, global_step=steps_done)
                    writer.add_scalar("Accuracy color", acc_color, global_step=steps_done)
                    writer.add_scalar("Accuracy seniority", acc_seniority, global_step=steps_done)
                    writer.add_scalar("Accuracy size", acc_size, global_step=steps_done)

                    test_accs = np.append(test_accs, acc_total)
                    # Early stopping
                    if test_accs.size > iterations_before_early_stopping\
                            and np.sum(test_accs[-earlystopping] < test_accs[-earlystopping:]) == 0:
                        print("Early stopping with accuracy {}".format(acc_total))
                        break

            if steps_done > n_iterations:
                    break

        if steps_done > n_iterations:
            break
