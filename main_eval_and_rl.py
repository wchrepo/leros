from datetime import datetime
import itertools
import json
import logging
import math
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import argparse
from utils import ensure_dir, ceil_div, set_seed
from data import QADataset,QADataset_WithK
from rl.policy import Policy
from rl.value import Value
from rl.reward import Reward
from eval_and_rl import PPOTrainer

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument(
        '--mode', type=str, choices=['train', 'eval'], required=True, help='train or eval?')
    parser.add_argument('--experiment_name',type=str,default='CWWV_STAGE3')
    # dataset
    parser.add_argument(
        '--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument(
        '--eval_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg,numersense,riddlesense,quartz,hellaswag')
    parser.add_argument(
        '--eval_givenk', type=str, default=None
    )
    parser.add_argument(
        '--eval_split', type=str, default='dev', choices=['train','dev', 'test'])

    # model
    parser.add_argument(
        '--model_type', type=str, default='t5-large', help='model used for policy, ref policy, and value')
    parser.add_argument(
        '--model_ckpt', type=str, default=None, help='model ckpt used for policy and ref policy (NOT value!)')
    parser.add_argument(
        '--use_model_ckpt_for_value', action='store_true', default=False)
    parser.add_argument(
        '--policy_value_sharing', action='store_true', default=False)
    parser.add_argument(
        '--qa_model_type', type=str, default='allenai/unifiedqa-t5-large', help='model used for QA')
    parser.add_argument(
        '--qa_model_ckpt', type=str, default=None, help='model ckpt used for QA')
    parser.add_argument(
        '--max_input_len', type=int, default=256, help='max length of the input prompt')
    parser.add_argument(
        '--max_output_len', type=int, default=128, help='max length of the output knowledge')
    parser.add_argument(
        '--load_from_ckpt', type=str, default=None, help='ckpt path to resume training or run eval')
    parser.add_argument(
        '--load_pretrained_ckpt', type=str, default=None, help='ckpt path to resume training or run eval')
    parser.add_argument(
        '--eval_ckpt', type=str, default=None, help='rainier ckpt to run eval')

    # reward
    parser.add_argument(
        '--kl_coef', type=float, default=0.2, help='coefficient for KL term in reward')
    parser.add_argument(
        '--reward_shape', type=int, default=4, help='refer to reward.py for implementation of each option')
    parser.add_argument(
        '--gain', type=float, default=None, help='precomputed normalization factor for reward')
    parser.add_argument(
        '--bias', type=float, default=None, help='precomputed normalization factor for reward')

    # ppo
    parser.add_argument(
        '--pg_coef', type=float, default=1.0, help='policy loss coefficient')
    parser.add_argument(
        '--vf_coef', type=float, default=1.0, help='value loss coefficient')
    parser.add_argument(
        '--cliprange', type=float, default=.2, help='clip parameter for policy gradient')
    parser.add_argument(
        '--cliprange_value', type=float, default=.2, help='clip parameter for value function')
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='discount factor for rewards')
    parser.add_argument(
        '--lam', type=float, default=0.95, help='lambda parameter for generalized advantage estimation')
    parser.add_argument(
        '--whiten_rewards', action='store_false', default=True, help='whether to normalize reward in each minibatch')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # train
    parser.add_argument(
        '--total_episodes', type=int, default=1000000, help='total number of episodes')
    parser.add_argument(
        '--num_warmup_step_ratio', type=float, default=0.0, help = 'ratio of number of steps to use for warmup with linear warmup')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--noptepochs', type=int, default=4, help='number of ppo epochs reusing rollouts')
    parser.add_argument(
        '--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='temperature for sampling from policy during training')

    # eval
    parser.add_argument(
        '--num_samples', type=int, default=10, help='number of knowledges to sample during eval')
    parser.add_argument(
        '--top_p', type=float, default=0.5, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--ensembling', type=str, default='max', choices=['max', 'moe', 'poe', 'majority'], help='ensembling method for inference')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval', type=int, default=1, help='step interval to print out logs')
    parser.add_argument(
        '--save_interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--nosave', default=False, action='store_true')
    parser.add_argument(
        '--skip_first_valid', default=False, action='store_true')
    parser.add_argument(
        '--eval_baseline', action='store_true', help='whether to evaluate the no-knowledge baseline')
    parser.add_argument(
        '--cuda_deterministic', action='store_false', default=True,
        help='sets flags for determinism when using CUDA (potentially slow!)')
    parser.add_argument(
        '--max_valid_batches', type=int,default=300
    )
    parser.add_argument(
        '--max_reward_norm_batches', type=int,default=2000
    )
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    set_seed(args.seed, args.cuda_deterministic)

    # GPUs
    num_gpus = torch.cuda.device_count()
    log.info(f'Detected {num_gpus} GPUS')
    devices = {}
    if torch.cuda.is_available():
        for i in range(num_gpus):
            devices[i] = torch.device('cuda:' + str(i))
    else:
        devices[0] = torch.device('cpu')

    # device_map = None
    # if args.mode == 'train':
    #     if num_gpus == 8:  # 8x RTX6000
    #         device_map = {
    #             0: [0],
    #             1: [1, 2, 3],
    #             2: [4, 5, 6],
    #             3: [7, 8, 9],
    #             4: [10, 11, 12],
    #             5: [13, 14, 15],
    #             6: [16, 17, 18, 19],
    #             7: [20, 21, 22, 23],
    #         }
    #     else:
    #         log.error('Invalid number of GPUs! Please use 8')
    #         exit(-1)
    # elif args.mode == 'eval':
    #     if num_gpus == 4:  # 4x RTX6000
    #         device_map = {
    #             0: [0],
    #             1: [1, 2, 3, 4, 5, 6, 7],
    #             2: [8, 9, 10, 11, 12, 13, 14, 15],
    #             3: [16, 17, 18, 19, 20, 21, 22, 23],
    #         }
    if num_gpus == 4:
        device_policy = "cuda:1"
        # device_map_policy = None
        device_map_policy = {
            1:[0,1,2,3,4,5,6,7,8,9,10,11],
            2:[12,13,14,15,16,17,18,19,20,21,22,23],
        }
        device_value = "cuda:3"
        device_map_value = None
        # device_map_value = {
        #     0:[0],
        #     3:[1,2,3,4,5,6,7,8,9,10,11],
        #     4:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_ref_policy = "cuda:0"
        device_qa_model = "cuda:0"
    elif num_gpus == 3:
        device_policy = "cuda:1"
        device_map_policy = None
        # device_map_policy = {
        #     1:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     2:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_value = "cuda:2"
        device_map_value = None
        # device_map_value = {
        #     3:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     4:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_ref_policy = "cuda:0"
        device_qa_model = "cuda:0"
    elif num_gpus == 2:
        device_policy = "cuda:1"
        device_map_policy = None
        # device_map_policy = {
        #     1:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     2:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_value = "cuda:0"
        device_map_value = None
        # device_map_value = {
        #     3:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     4:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_ref_policy = "cuda:0"
        device_qa_model = "cuda:1"
    elif num_gpus == 1:
        device_policy = "cuda:0"
        device_map_policy = None
        # device_map_policy = {
        #     1:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     2:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_value = "cuda:0"
        device_map_value = None
        # device_map_value = {
        #     3:[0,1,2,3,4,5,6,7,8,9,10,11],
        #     4:[12,13,14,15,16,17,18,19,20,21,22,23],
        # }
        device_ref_policy = "cuda:0"
        device_qa_model = "cuda:0"
    elif num_gpus == 5:
        device_policy = "cuda:1"
        # device_map_policy = None
        device_map_policy = {
            1:[0,1,2,3,4,5,6,7,8,9,10,11],
            2:[12,13,14,15,16,17,18,19,20,21,22,23],
        }
        device_value = "cuda:3"
        device_map_value = {
            3:[0,1,2,3,4,5,6,7,8,9,10,11],
            4:[12,13,14,15,16,17,18,19,20,21,22,23],
        }
        device_ref_policy = "cuda:0"
        device_qa_model = "cuda:0"
    elif num_gpus == 6:
        device_policy = "cuda:1"
        # device_map_policy = None
        device_map_policy = {
            1:[0,1,2,3,4,5,6,7,8,9,10,11],
            2:[12,13,14,15,16,17,18,19,20,21,22,23],
        }
        device_value = "cuda:3"
        # device_map_value = None
        device_map_value = {
            3:[0,1,2,3,4,5,6,7,8,9,10,11],
            4:[12,13,14,15,16,17,18,19,20,21,22,23],
        }
        device_ref_policy = "cuda:5"
        device_qa_model = "cuda:0"
    # Set up save directories
    if not args.nosave:
        if args.mode == 'train':
            args.output_dir = f'log/{args.experiment_name}/{args.model_type}/'
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.run_name = args.save_dir.split('/')[-1]
                with open(os.path.join(args.save_dir, 'args.json')) as f:
                    args.__dict__.update(json.load(f))
            elif args.model_ckpt is not None:
                time = datetime.now()
                args.run_name = os.path.dirname(os.path.dirname(args.model_ckpt)).split('/')[-1]+'_'+time.strftime('%b%d_%H-%M-%S')
                args.save_dir = os.path.join(args.output_dir, args.run_name)
            else:
                time = datetime.now()
                date_time = time.strftime('%b%d_%H-%M-%S')
                import socket
                args.run_name = date_time + '_' + socket.gethostname()
                args.save_dir = os.path.join(args.output_dir, args.run_name)
            args.reward_dir = os.path.join(args.save_dir, 'reward')
            args.model_dir = os.path.join(args.save_dir, 'model')
            # args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
            args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
            args.inference_dir = os.path.join(args.save_dir, 'inference')
            for d in [args.save_dir, args.reward_dir, args.model_dir,  args.knowledge_dir, args.inference_dir]:
                ensure_dir(d)

        elif args.mode == 'eval':
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.save_dir = args.save_dir.replace('log/', 'eval/')
                ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}-{args.eval_split}-{args.top_p}'
            elif args.eval_ckpt is not None:
                args.save_dir = "eval/CKPT/"+os.path.splitext(os.path.basename(args.eval_ckpt))[0]+f'-{args.eval_split}-{args.top_p}'
            elif args.model_ckpt is not None:
                args.save_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_ckpt)))
                args.save_dir = args.save_dir.replace('log/', 'eval/')
                ckp = args.model_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}-{args.eval_split}-{args.top_p}'
            elif args.eval_givenk:
                args.save_dir = "eval/givenk/"+args.eval_givenk.replace(',','__')
            elif args.qa_model_ckpt and args.eval_baseline:
                args.save_dir = "eval/qabaseline/"
            else:
                log.error('You must provide either --ckpt or --load_from_ckpt!')
                exit(-1)
            args.save_dir += f'-{"_".join(args.qa_model_ckpt.split("/")[-3:])}' if args.qa_model_ckpt else args.qa_model_type.split('/')[-1]
            args.run_name = args.save_dir.split('/')[-1]
            # args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
            args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
            args.inference_dir = os.path.join(args.save_dir, 'inference')
            for d in [args.save_dir,  args.knowledge_dir, args.inference_dir]:
                ensure_dir(d)

        log.info(f'Write to output directory: {args.save_dir}')
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Load data
    log.info(f'Loading data ...')

    if args.mode == 'train':
        train_dataset = QADataset('train', args.train_tasks)
        # train ds is shuffled in its constructor
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded train set with {len(train_dataset)} instances')

        eval_dataset = QADataset('dev', args.train_tasks)
        import random
        random.Random(1).shuffle(eval_dataset.instances)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded dev set with {len(eval_dataset)} instances')

    elif args.mode == 'eval':
        train_dataset = None
        train_dataloader = None
        if args.eval_givenk:
            eval_dataset = QADataset_WithK(args.eval_split,args.eval_givenk)
        else:
            eval_dataset = QADataset(args.eval_split, args.eval_tasks)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded {args.eval_split} set with {len(eval_dataset)} instances')


    # Initialize models and optimizer
    log.info(f'Initializing models ...')

    if args.mode == 'train':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            policy_value_sharing=args.policy_value_sharing,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            device=device_ref_policy,
            device_map=None,
        )
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            policy_value_sharing=args.policy_value_sharing,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            device=device_policy,
            device_map=device_map_policy,
        )
        # TODO: Try initializing this with model_ckpt as well
        value = Value(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt if args.use_model_ckpt_for_value else None,
            model=policy.model if args.policy_value_sharing else None,
            device=device_value,
            device_map=device_map_value,
        )
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            device=device_qa_model,
        )

        optimizer = torch.optim.Adam(policy.model.parameters() if args.policy_value_sharing else itertools.chain(policy.model.parameters(), value.model.parameters()), lr=args.lr, eps=1e-5)
        args.total_steps = ceil_div(args.total_episodes, args.batch_size)
        warmup_steps = math.ceil(args.num_warmup_step_ratio * args.total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.total_steps)
        init_step = 0
        eval_accs = {}

        # Load from checkpoint if continue training
        if args.load_pretrained_ckpt is not None:
            checkpoint = torch.load(args.load_pretrained_ckpt)
            log.info(f'Initializing from pretrained ckpt')
            ref_policy.model.load_state_dict(checkpoint['policy_model'])
            policy.model.load_state_dict(checkpoint['policy_model'])
            checkpoint.clear()
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt)
            policy.model.load_state_dict(checkpoint['policy_model'])
            value.model.load_state_dict(checkpoint['value_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            init_step = checkpoint['step']
            eval_accs = checkpoint['eval_accs']
            checkpoint.clear()

            # Reuse the reward normalization results
            reward.read_reward_norm(args.reward_dir)

    elif args.mode == 'eval':
        ref_policy = None
        if args.eval_givenk:
            policy = None
        else:
            policy = Policy(
                model_type=args.model_type,
                model_ckpt=args.model_ckpt,
                policy_value_sharing=args.policy_value_sharing,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                device_map=device_map_policy,
                device=device_policy,
            )
        value = None
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            device=device_qa_model,
        )

        optimizer = None
        scheduler = None
        init_step = 0
        eval_accs = {}

        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location=torch.device('cpu'))
            policy.model.load_state_dict(checkpoint['policy_model'])
            init_step = checkpoint['step']
            checkpoint.clear()
        elif args.eval_ckpt is not None:
            checkpoint = torch.load(args.eval_ckpt, map_location=torch.device('cpu'))
            policy.model.load_state_dict(checkpoint)
            checkpoint.clear()
    

    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        policy_model=policy,
        ref_policy_model=ref_policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        init_step=init_step,
        eval_accs=eval_accs,
        log=log,
    )

    # Normalize the rewards to so that initially they have mean 0, var 1
    if args.mode == 'train':
        if args.load_from_ckpt is None:
            log.info('Setting reward norm')
            if args.gain is not None and args.bias is not None:
                reward.gain = args.gain
                reward.bias = args.bias
            else:
                trainer.set_reward_norm()
            log.info(f'Set reward norm as gain = {reward.gain}, bias = {reward.bias}')
            if not args.nosave:
                reward.write_reward_norm(args.reward_dir)

    # Evaluate baseline (no knowledge)
    if args.eval_baseline:
        trainer.eval(step=-1)
    # Train or evaluate
    elif args.mode == 'train':
        pbar = tqdm(list(range(init_step, args.total_steps + 1)))
        for step in pbar:
            trainer.train(step)
    elif args.mode == 'eval':
        if args.eval_givenk:
            trainer.eval(step=-1,givenk=True)
        else:
            trainer.eval(init_step)


if __name__ == '__main__':
    main()

