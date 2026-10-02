import argparse


from algorithms.secbad.utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--learner_type', default='secbad',
                        help="select from varibad, secbad, oracle_truncate")

    # file path

    parser.add_argument('--dataset_path', default='../data/dataset_2s_train/',
                        help="dataset path for training")

    parser.add_argument('--dataset_path_test', default='../data/dataset_2s_test/',
                        help="dataset path for testing")

    parser.add_argument('--network_traces_path', default='../data/network_traces/sampled_4G_train/',
                        help="network traces path for train")

    parser.add_argument('--network_traces_path_test', default='../data/network_traces/sampled_4G/',
                        help="network traces path for train")

    parser.add_argument('--chunklength', default=2000.,
                        help="length of each video chunk (ms)")

    # use wrong priori

    parser.add_argument('--use_invalid_action_mask', default=True,
                        help="use mask for invalid actions")

    parser.add_argument('--traj_len', type=int, default=100)

    # --- GENERAL ---

    parser.add_argument('--num_frames', type=int, default=2e6,
                        help='number of frames to train')
    # parser.add_argument('--max_rollouts_per_task', type=int,
    #                     default=1, help='number of MDP episodes for adaptation')  # 2 originally
    parser.add_argument('--exp_label', default='secbad',
                        help='label (typically name of method)')
    parser.add_argument(
        '--env_name', default='short_video_env', help='environment to train on')

    # --- POLICY ---
    # RL algorithm
    parser.add_argument('--policy', type=str, default='dqn',
                        help='choose: '
                             'a2c'
                             'ppo'
                             'dqn'
                             'sac')
    parser.add_argument('--policy_optimiser', type=str,
                        default='adam', help='choose: rmsprop, adam')

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument,
                        default=True, help='condition policy on state')
    parser.add_argument('--pass_latent_to_policy', type=boolean_argument,
                            default=True, help='condition policy on VAE latent')

    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--max_buffer_size', type=int, default=4,
                        help="max buffer size of videos")

    parser.add_argument('--state_embedding_dim', type=int, default=5)
    parser.add_argument('--encoder_state_embedding_dim', type=int, default=5)
    parser.add_argument('--latent_embedding_dim', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='dimensionality of latent space')

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_for_policy', type=boolean_argument,
                        default=False, help='normalise state input')
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument,
                        default=True, help='normalise rew for RL train')
    parser.add_argument('--norm_adv_for_policy', type=boolean_argument,
                        default=True, help='normalise rew for RL train')

    parser.add_argument('--norm_latent_for_policy', type=boolean_argument,
                        default=True, help='normalise latent input')

    parser.add_argument('--policy_entropy_coef', type=float,
                        default=0.2, help='entropy term coefficient')

    parser.add_argument('--policy_use_gae', type=boolean_argument, default=False,
                        help='use generalized advantage estimation')

    parser.add_argument('--policy_anneal_lr', type=boolean_argument,
                        default=True, help='anneal LR over time')
    parser.add_argument('--ppo_use_clipped_value_loss',
                        type=boolean_argument, default=False, help='clip value loss')

    parser.add_argument('--ppo_use_huberloss', type=boolean_argument,
                        default=False, help='use huberloss instead of MSE')

    parser.add_argument('--policy_tau', type=float,
                        default=0.95, help='gae parameter')

    parser.add_argument('--policy_layers', nargs='+', default=[128, 64])
    parser.add_argument('--policy_activation_function',
                        type=str, default='relu', help='tanh/relu/leaky-relu')

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int,
                        default=4, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=1,
                        help='number of minibatches to split the data')
    parser.add_argument('--ppo_clip_param', type=float,
                        default=0.1, help='clamp param')

    # DQN specific
    parser.add_argument('--target_update_interval', type=int,
                        default=20, help='clamp param')
    parser.add_argument('--epsilon_decay_steps', type=int,
                        default=1500, help='clamp param')
    parser.add_argument('--dqn_gamma', type=int,
                        default=0.95, help='clamp param')

    # SAC specific
    parser.add_argument('--actor_lr_sac', type=float,
                        default=1e-4, help='clamp param')
    parser.add_argument('--critic_lr_sac', type=float,
                        default=1e-3, help='clamp param')
    parser.add_argument('--alpha_lr_sac', type=float,
                        default=1e-3, help='clamp param')
    parser.add_argument('--target_entropy_sac', type=float,
                        default=1e-3, help='clamp param')
    parser.add_argument('--tau_sac', type=float,
                        default=0.005, help='clamp param')
    parser.add_argument('--gamma_sac', type=float,
                        default=0.99, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=1e-4,
                        help='learning rate (default: 7e-4)')
    # 并行训练的参数，指明有多少个子进程收集经验
    parser.add_argument('--num_processes', type=int, default=8,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    # 参数更新前收集的经验长度
    parser.add_argument('--policy_num_steps', type=int, default=100,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--policy_eps', type=float, default=1e-8,
                        help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_value_loss_coef', type=float,
                        default=0.1, help='value loss coefficient')
    parser.add_argument('--policy_gamma', type=float,
                        default=0.99, help='discount factor for rewards')

    # grad_clip
    parser.add_argument('--policy_max_grad_norm', type=float,
                        default=3., help='max norm of gradients')
    parser.add_argument('--encoder_max_grad_norm', type=float,
                        default=15., help='max norm of gradients')
    parser.add_argument('--decoder_max_grad_norm', type=float,
                        default=15., help='max norm of gradients')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--max_input_history_length', type=int, default=4,
                        help='the length of history decision used for best segment inference (default: 5)')

    parser.add_argument('--use_best_latent_selection', type=boolean_argument,
                                default=True, help='select best latent feature with max predicted reward')

    parser.add_argument('--lr_vae', type=float, default=1e-3)
    parser.add_argument('--size_vae_buffer', type=int, default=800,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--precollect_len', type=int, default=2000,
                        help='how many frames to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--vae_buffer_add_thresh', type=float, default=1.,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae_batch_num_trajs', type=int, default=10,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--num_vae_updates', type=int, default=4,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--vae_anneal_lr', type=boolean_argument,
                        default=False, help='anneal LR over time')

    # - encoder
    parser.add_argument('--encoder_input_feature', type=str, default='all',
                        help='choose: '
                             'bw'
                             'user_behavior'
                             'all')

    parser.add_argument('--encoder_type', type=str, default='lstm',
                        help='choose: '
                             'rnn'
                             'gru'
                             'lstm'
                             'transformer')

    parser.add_argument('--encoder_layers_before_gru',
                        nargs='+', type=int, default=[])
    parser.add_argument('--encoder_gru_hidden_size', type=int,
                        default=64, help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder_layers_after_gru',
                        nargs='+', type=int, default=[])

    # - decoder

    parser.add_argument('--decode_reward', type=boolean_argument,
                        default=True, help='use reward decoder')

    # - decoder: rewards
    parser.add_argument('--pass_norm_rew_to_vae', type=boolean_argument,
                        default=False, help='use norm rew for rew pred')
    parser.add_argument('--input_state', type=boolean_argument,
                        default=True, help='use prev state for rew pred')
    parser.add_argument('--input_latent', type=boolean_argument,
                        default=True, help='use prev action for rew pred')
    parser.add_argument('--reward_decoder_layers',
                        nargs='+', type=int, default=[128, 64])
    parser.add_argument('--rew_pred_type', type=str, default='deterministic',
                        help='choose: '
                             'gaussian (predict p(r|s))'
                             'deterministic (treat as regression problem)')

    # --- ABLATIONS ---

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=100,
                        help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models',
                        type=boolean_argument, default=True, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='eval interval, one eval per n updates')
    parser.add_argument('--results_log_dir', default=None,
                        help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--seed',  nargs='+', type=int,
                        default=[12])  # 12, 66, 95

    return parser.parse_args(rest_args)
