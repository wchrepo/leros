#把PPOTrainer的Eval功能集成进来。
import argparse
from datetime import datetime
import json
import logging
import numpy as np
import os
import random
from tqdm.auto import tqdm
from ppo import PPOTrainer

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers.optimization import get_constant_schedule_with_warmup,get_linear_schedule_with_warmup,get_constant_schedule
# import wandb
import mlflow

from data import PERSON_NAMES
from utils import safe_cfg, ensure_dir, set_seed

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)

# import re
# personx_regex = re.compile(r'\b[Pp]erson[Xx]\b|\b[Pp]erson [Xx]\b|\b[Xx]\b')
# persony_regex = re.compile(r'\b[Pp]erson[Yy]\b|\b[Pp]erson [Yy]\b|\b[Yy]\b')
# personz_regex = re.compile(r'\b[Pp]erson[Zz]\b|\b[Pp]erson [Zz]\b|\b[Zz]\b')
{'source': 'PersonX treats PersonY with respect. What is the intent of PersonX? \\n (A) to rest his feet (B) to be separate from PersonY (C) thinks highly of a person','target': "As a result of PersonX treats PersonY with respect, PersonX would feel respectful. PersonX can be seen as respectful because PersonX awaits PersonY's turn. As a result of PersonX awaits PersonY's turn, PersonY thinks."}
def give_random_name(d):
    names = random.sample(PERSON_NAMES,3)
    if "PersonX" in d['source']: #
        d['source'] = d['source'].replace("PersonX",names[0])
        d['target'] = d['target'].replace("PersonX",names[0])
    if "PersonY" in d['source']:
        d['source'] = d['source'].replace("PersonY",names[1])
        d['target'] = d['target'].replace("PersonY",names[1])
    if "PersonZ" in d['source']:
        d['source'] = d['source'].replace("PersonZ",names[2])
        d['target'] = d['target'].replace("PersonZ",names[2])
    return d

class Ds(Dataset):
    def __init__(self, paths, split,score_threshold=None,random_naming=None,max_train_instances=None,max_eval_instances=None,top_n=3):
        super().__init__()
        self.ds = self.init_ds(paths, split,score_threshold,random_naming,max_train_instances,max_eval_instances,top_n)

    def init_ds(self, paths, split,score_threshold=None,random_naming=None,max_train_instances=None,max_eval_instances=None,top_n=3):
        ds = []
        for path,datasettype in paths:
            with open(path) as f:
                js = json.load(f)
            if datasettype == 'syn':
                for item in js:
                    #
                    targets = []
                    if 'scores' in item:
                        for k in range(len(item['knowledges'])):
                            if score_threshold is None or item['scores'][k]>=score_threshold:
                                targets.append((item['scores'][k],item['knowledges'][k]))
                                # for evaluation, only keep the first knowledge for sake of speed
                                if split == 'eval':
                                    break
                            targets.sort(reverse=True)
                            targets=targets[:top_n]
                        targets = [target[1] for target in targets]
                    else:
                        targets = item['knowledges'][:top_n]
                    for target in targets:
                        d = {'source': item['query'].replace('\n',' \\n '), 'target': target}
                        if random_naming is not None and random.random()<random_naming:
                            d = give_random_name(d)
                        ds.append(d)
            elif datasettype == 'real':
                for item in js:
                    for k in range(len(item['knowledges'])):
                        ds.append({'source': item['query'], 'target': item['knowledges'][k]})
                        # for evaluation, only keep the first knowledge for sake of speed
                        if split == 'eval':
                            break
            elif datasettype == 'gen_qka':
                for d in js:
                    if 'qka' in args.gen_dataset_kstrategy:
                        #
                        if 'knowledge' in d and d['cs'] and (score_threshold is None or d['scores'][0]>=score_threshold):
                            ds.append({'source':d['question'].replace('\n',' \\n '),'target':d['knowledge']})
                        elif 'answers' in d:
                            for i,a in enumerate(d['answers']):
                                if a['cs'] and (score_threshold is None or d['scores'][i]>=score_threshold):
                                    ds.append({'source':d['question'].replace('\n',' \\n '),'target':a['knowledge']})
                    if 'kq' in args.gen_dataset_kstrategy and d['cs'] and (score_threshold is None or d['scores'][-1]>=score_threshold):
                        ds.append({'source':d['question'].replace('\n',' \\n '),'target':d['KQknowledge']})

            # for item in js:
            #     for k in range(len(item['knowledges'])):
            #         d = {'source': item['query'].replace('\n',' \\n '), 'target': item['knowledges'][k]}
            #         if random_naming is not None and random.random()<random_naming:
            #             d = give_random_name(d)
            #         if score_threshold is None or item['scores'][k]>=score_threshold:
            #             ds.append(d)
            #             # for evaluation, only keep the first knowledge for sake of speed
            #             if split == 'eval':
            #                 break
        if split == 'train':
            random.shuffle(ds)
            if isinstance(max_train_instances,float):
                max_train_instances = int(max_train_instances*len(ds))
            ds = ds[:max_train_instances]
        if split == 'eval':
            random.shuffle(ds)
            ds = ds[:max_eval_instances]
        if is_main_process:
            print(f'{split} set size = {len(ds)}')
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    @staticmethod
    def collate_fn(batch):
        return {k: [item[k] for item in batch] for k in batch[0]}

class Trainer:
    def __init__(self,
                 args,
                 train_dataloader,
                 eval_dataloader,
                 tokenizer,
                 model,
                 optimizer,
                 scheduler,
                 init_step,
                 eval_losses,
                 device,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader)
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        if is_main_process and not args.nosave:
            # self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            mlflow_is_resume = "MLFLOW_RUN_ID" in os.environ
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(run_name=args.run_name)
            if not mlflow_is_resume:
                try:
                    mlflow.log_params(safe_cfg(vars(args)))
                except Exception as e:
                    print(e)
            # wandb.define_metric('train/step')
            # wandb.define_metric('eval/step')
            # wandb.define_metric('train/loss', step_metric='train/step', summary='min')
            # wandb.define_metric('eval/loss', step_metric='eval/step', summary='min')

        self.train_sampler = iter(self.train_dataloader)
        for _ in range((init_step * args.accumulate_grad_batches) % len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_losses = eval_losses

    def loss(self, batch):
        source_tok = self.tokenizer.batch_encode_plus(
            batch['source'],
            return_tensors='pt', padding=True, truncation='longest_first', max_length=self.args.max_input_len).to(self.device)
        target_tok = self.tokenizer.batch_encode_plus(
            batch['target'],
            return_tensors='pt', padding=True, truncation='longest_first', max_length=self.args.max_output_len).to(self.device)
        labels = target_tok.input_ids
        labels[target_tok.attention_mask == 0] = -100

        loss = self.model(
            input_ids=source_tok.input_ids,
            attention_mask=source_tok.attention_mask,
            labels=labels,
        ).loss

        return loss

    def train(self, step):
        if is_main_process:
            self.eval(step=step)
            self.save(step=step)
        batch_loss = torch.tensor(0.,device=self.device)
        self.model.train()
        self.optimizer.zero_grad()
        for _ in range(self.args.accumulate_grad_batches):
            try:
                batch = next(self.train_sampler)
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch = next(self.train_sampler)
            loss = self.loss(batch)/self.args.accumulate_grad_batches
            loss.backward()
            batch_loss += loss.item()
        self.optimizer.step()
        self.scheduler.step()
        if ddp:
        # loss = loss.detach()
            losses = [torch.zeros_like(batch_loss) for i in range(self.args.world_size)]
            torch.distributed.all_gather(tensor_list=losses,tensor=batch_loss)
            batch_loss = torch.stack(losses).mean()
        if is_main_process and not self.args.nosave:
            if step % self.args.log_interval == 0:
                # self.writer.add_scalar('train/loss', loss.item(), step)
                # wandb.log({'train/loss': loss.item(), 'train/step': step})
                mlflow.log_metric('train/loss',batch_loss.item(),step)

    def eval(self, step):
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_losses:
            return
        log.info(f'Evaluating [step {step}] ...')

        losses = []
        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            self.model.eval()
            with torch.no_grad():
                loss = self.loss(batch)
            losses.append(loss.item())
        loss = np.mean(losses)

        if self.args.nosave:
            self.eval_losses[step] = loss
        else:
            # self.writer.add_scalar('eval/loss', loss, step)
            # wandb.log({'eval/loss': loss.item(), 'eval/step': step})
            mlflow.log_metric('eval/loss',loss.item(),step)

            prev_best_step = None if len(self.eval_losses) == 0 else min(self.eval_losses, key=self.eval_losses.get)
            self.eval_losses[step] = loss
            if prev_best_step is None or loss < self.eval_losses[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                self.save(step, last=False)
                log.info(f'Best ckpt updated to [step {step}]')

    

    def save(self, step, last=True):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict(),
            'step': step,
            'eval_losses': self.eval_losses,
        }, f'{self.args.model_dir}/{"last" if last else "ckp_" + str(step)}.pth')
        
        log.info(f'[step {step}] model checkpoint saved')
    
    

def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--experiment_name',type=str,default='ATOMIC_CON_STAGE1')
    parser.add_argument('--run_name',type=str,default=None)
    parser.add_argument('--model_type', type=str, default='t5-large')
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--model_ckpt',type=str,default=None)
    parser.add_argument('--max_input_len', type=int, default=192)
    parser.add_argument('--max_output_len', type=int, default=128)
    parser.add_argument('--keep_top_n', type=int, default=3)
    # train
    parser.add_argument('--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument('--ckg_dataset_paths', type=str, nargs='*', default=[])
    parser.add_argument('--gen_dataset_paths', type=str, nargs='*', default=['data/auto/gen_data_v1'])
    parser.add_argument('--gen_dataset_kstrategy',type=str,default='qka') #
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_scheduler', type=str, default="linear")
    parser.add_argument('--warm_ups', type=int, default=100)
    

    # other
    parser.add_argument(
        '--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument(
        '--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--nosave', default=False, action='store_true')

    parser.add_argument('--score_threshold',default=None,type=float)
    parser.add_argument('--random_naming',default=None,type=float)
    parser.add_argument('--max_eval_instances',default=5000,type=int)
    parser.add_argument('--max_train_instances', type=float, default=None)


    args = parser.parse_args()
    return args

def main():
    global is_main_process,tqdm,ddp,args
    args = get_args()

    #distributed
    if 'LOCAL_RANK' in os.environ:
        ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        is_main_process = local_rank == 0
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        args.world_size = torch.distributed.get_world_size()
        log.info(f"Launch ddp process {local_rank}")
    else:
        ddp = False
        is_main_process = True
        args.world_size = 1

    set_seed()


    # Set up save directories
    if not args.nosave and is_main_process:
        
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
            args.run_name = args.run_name or args.save_dir.split('/')[-1]
        else:
            args.output_dir = f'log/{args.experiment_name}/{args.model_type}/'
            time = datetime.now()
            date_time = time.strftime('%b%d_%H-%M-%S')
            import socket
            args.run_name = args.run_name or date_time + '_' + socket.gethostname()
            args.save_dir = os.path.join(args.output_dir, args.run_name)
        args.model_dir = os.path.join(args.save_dir, 'model')
        # args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        for d in [args.save_dir, args.model_dir]:
            ensure_dir(d)
    
        log.info(f'Write to output directory: {args.save_dir}')
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Load data
    # log.info(f'Loading data ...')
    if args.ckg_dataset_paths:
        if args.score_threshold is not None:
            train_paths = [(os.path.join(path,"train.json.scored"),'syn')for path in args.ckg_dataset_paths]
            eval_paths = [(os.path.join(path,"dev.json.scored"),'syn') for path in args.ckg_dataset_paths]
        else:
            train_paths = [(os.path.join(path,"train.json"),'syn') for path in args.ckg_dataset_paths]
            eval_paths = [(os.path.join(path,"dev.json"),'syn') for path in args.ckg_dataset_paths]
    else:
        train_paths = []
        eval_paths = []
    if args.train_tasks:
        train_tasks = args.train_tasks.split(',')
        train_paths.extend([(f'data/knowledge/GKP/knowledge/knowledge_gkp_gpt3curie.train.{task}.json','real') for task in train_tasks])
        eval_paths.extend([(f'data/knowledge/GKP/knowledge/knowledge_gkp_gpt3curie.dev.{task}.json','real') for task in train_tasks])
    if args.gen_dataset_paths:
        if args.score_threshold is not None:
            train_paths.extend([(f'{gen_path}_train.json.scored','gen_qka') for gen_path in args.gen_dataset_paths])
            eval_paths.extend([(f'{gen_path}_dev.json.scored','gen_qka') for gen_path in args.gen_dataset_paths])
        else:
            train_paths.extend([(f'{gen_path}_train.json','gen_qka') for gen_path in args.gen_dataset_paths])
            eval_paths.extend([(f'{gen_path}_dev.json','gen_qka') for gen_path in args.gen_dataset_paths])
    print(train_paths)
    print(eval_paths)
    train_dataset = Ds(train_paths, 'train',args.score_threshold,args.random_naming,max_train_instances=args.max_train_instances,top_n=args.keep_top_n)
    eval_dataset = Ds(eval_paths, 'eval',args.score_threshold,args.random_naming,args.max_eval_instances)
    if is_main_process:
        import rich
        rich.print("Sample of train data")
        rich.print(train_dataset[:10])
    # distributed
    node_batch_size = args.batch_size//args.accumulate_grad_batches
    train_sampler = None
    
    if ddp:
        assert node_batch_size%args.world_size == 0
        node_batch_size = node_batch_size//args.world_size
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=node_batch_size,sampler=train_sampler,
        shuffle=False, drop_last=True, collate_fn=Ds.collate_fn)
    if is_main_process:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
            batch_size=node_batch_size,
            shuffle=False, drop_last=False, collate_fn=Ds.collate_fn)
    else:
        eval_dataloader = None
    # train ds is shuffled in its constructor
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=Ds.collate_fn)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=Ds.collate_fn)

    # Initialize models and optimizer
    # log.info(f'Initializing models ...')
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type)
    model.to("cuda")
    model_ = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,args.warm_ups,args.total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer,args.warm_ups)
    init_step = 0
    eval_losses = {}

    # Load from checkpoint if continue training
    if args.load_from_ckpt is not None:
        checkpoint = torch.load(args.load_from_ckpt,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_step = checkpoint['step']
        eval_losses = checkpoint['eval_losses']
        checkpoint.clear()
    elif args.model_ckpt:
        print("Loading model_ckpt")
        checkpoint = torch.load(args.model_ckpt,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        checkpoint.clear()

   
    # Set up trainer
    trainer = Trainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        init_step=init_step,
        eval_losses=eval_losses,
        device="cuda",
    )

    # reward = Reward(
    #     model_type=args.qa_model_type,
    #     model_ckpt=args.qa_model_ckpt,
    #     max_input_len=args.max_input_len,
    #     batch_size=args.batch_size,
    #     reward_shape=args.reward_shape,
    #     kl_coef=args.kl_coef,
    #     ensembling=args.ensembling,
    #     device="cuda:1",
    # )

    # Train
    if is_main_process:
        pbar = tqdm(list(range(init_step, args.total_steps + 1)),dynamic_ncols=True)
    else:
        pbar = iter(list(range(init_step, args.total_steps + 1)))
    for step in pbar:
        trainer.train(step)


if __name__ == '__main__':
    main()

