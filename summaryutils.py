import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def summary_trajectory(env, writer, n_actions, action_names, policy_net, state, observation, t, episode):
    image = env.render("rgb_array")
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Actions")
    ax1.bar(range(n_actions), policy_net(state).cpu().detach().numpy().reshape(-1))
    ax1.set_xticks(range(n_actions))
    ax1.set_xticklabels(action_names, fontdict=None, minor=False)
    ax1.set_ylabel("Q values")
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.set_title("{}".format(observation["mission"]))
    ax2.imshow(image)
    writer.add_figure("Q values episode {}".format(episode + 1), fig, global_step=t)


def summary_mean_max_q(dict_agent, memory, policy_net, writer, steps_done):
    # Sample a batch of states and compute mean and max values of Q
    if dict_agent["agent"] == "dqn-per" or dict_agent["agent"] == "double-dqn-per":
        transitions, is_weights, transition_idxs = memory.sample(dict_agent["batch_size"])
    else:
        transitions = memory.sample(dict_agent["batch_size"])
    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.cat(batch_transitions.curr_state)
    batch_mission = torch.cat(batch_transitions.mission)
    batch_curr_state_dict = {
        "image": batch_curr_state,
        "mission": batch_mission
    }
    q_values = policy_net(batch_curr_state_dict).detach()
    writer.add_scalar("mean Q", torch.mean(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)
    writer.add_scalar("max Q", torch.max(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)


def mission_tokenizer(dict_env, target, dim_tokenizer):
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():

        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())

        mission_color_onehot = torch.zeros(num_colors)
        if num_colors > 1:
            mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1
        mission_type_onehot = torch.zeros(num_types)
        mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

        return torch.cat((mission_color_onehot, mission_type_onehot))[None, :]
    else:
        # Constant mission
        return torch.zeros(1, dim_tokenizer)