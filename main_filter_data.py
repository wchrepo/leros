import argparse
import json
import os
from rl.reward import Reward
from tqdm.auto import tqdm

"""
  {
    "knowledge": "The first step to solve a problem is to identify possible solutions.",
    "question": "If PersonX needs to solve a problem, what is the first step they should take? \n(A) Stop and do nothing (B) Ask someone else to solve it (C) Create a plan of action (D) Brainstorm possible solutions",
    "choices": [
      "Stop and do nothing",
      "Ask someone else to solve it",
      "Create a plan of action",
      "Brainstorm possible solutions"
    ],
    "answer": "Brainstorm possible solutions",
    "answer_id": 3,
    "has_answer": true,
    "KQanswer_id": 3,
    "KQknowledge": "Before solving the problem, PersonX needed to brainstorm.",
    "cs": true
  },
"""

def normalize_for_uqa(data):
    for d in data:
        d['questions'] = d['questions'].replace('\n',' \\n ')
        # d['choicess'] = [choice[4:] for choice in d['choicess']]


# def get_scores(data,reward_model,auto_remove_duplicates=True,batch_size=32):
#     # normalize_for_uqa(data)
#     if auto_remove_duplicates:
#         for d in data:
#             d['knowledges'] = list(set(d['knowledges']))
#     flat_lengths = [len(d['knowledges']) for d in data]
#     flat_data = [
#         {
#             "questions":d["query"],"choicess":d["choices"],"answer_ixs":d["answer_id"],
#             "knowledges":k,
#         } 
#         for d in data for k in d['knowledges']
#     ]
#     normalize_for_uqa(flat_data)
#     batchified = batchify(flat_data,batch_size)
#     flat_scores = []
#     flat_answer_probs = []
#     for batch in tqdm(batchified):
#         results = reward_model.get_reward(**batch,override_gain=1,override_bias=0)
#         flat_scores.extend(results['rewards/raw'])
#         flat_answer_probs.extend(results['answer_probs'].tolist())
#     start = 0
#     for i in range(len(data)):
#         data[i]['scores'] = flat_scores[start:start+flat_lengths[i]]
#         data[i]['probs'] = flat_answer_probs[start:start+flat_lengths[i]]
#         start += flat_lengths[i]
#     return data

def get_flat_data(data,keys = ['answers','knowledge','KQknowledge']):
    # flat_lengths = [len([k for k in keys if k in d]) for d in data]
    flat_lengths = []
    flat_data = []
    for d in data:
        flat_length = 0
        for key in keys:
            if key in d:
                if isinstance(d[key],str):
                    flat_data.append({
                        "questions":d["question"],"choicess":d["choices"],"answer_ixs":d["KQanswer_id"],
                        "knowledges":d[key],
                    })
                    flat_length+=1
                else:
                    for a in d[key]:
                        flat_data.append({
                            "questions":d["question"],"choicess":d["choices"],"answer_ixs":d["KQanswer_id"] if "KQanswer_id" in d else d['answer_id'],
                            "knowledges":a['knowledge'],
                        })
                        flat_length+=1
        flat_lengths.append(flat_length)
    # flat_data = [
    #     {
    #         "questions":d["question"],"choicess":d["choices"],"answer_ixs":d["answer_id"],
    #         "knowledges":d[key],
    #     } 
    #     for d in data for key in keys if key in d
    # ]
    normalize_for_uqa(flat_data)
    return flat_data,flat_lengths

def get_scores_for_batch(batch,reward_model):
    results = reward_model.get_reward(**batch,override_gain=1,override_bias=0)
    return results['rewards/raw'],results['answer_probs'].tolist()

def batchify(data,batch_size=32):
    start = 0
    all_keys = data[0].keys()
    results = []
    while start < len(data):
        batched_dict = {}
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in data[start:start+batch_size]:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        # yield batched_dict
        results.append(batched_dict)
        start += batch_size
    return results
    

def filter_data(data, keep_threshold=0.,keep_ratio=1):
    all_scores = [s for d in data for s in d['scores']]
    all_scores.sort(reverse=True)
    keep_ratio_idx = int(len(all_scores) * keep_ratio)-1
    if keep_ratio_idx>=0 and keep_ratio_idx<len(all_scores):
        keep_threshold = max(keep_threshold,all_scores[keep_ratio_idx])
    results = []
    for d in data:
        result_d = d.copy()
        result_d['knowledges'] = []
        result_d['scores'] = []
        result_d['probs'] = []
        for i in range(len(d['knowledges'])):
            if d['scores'][i]>=keep_threshold:
                result_d['knowledges'].append(d['knowledges'][i])
                result_d['scores'].append(d['scores'][i])
                result_d['probs'].append(d['probs'][i])
        if len(result_d['knowledges'])>0:
            results.append(result_d)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument(
            '--keep_threshold', type=float, default=0.0, help='Keep threshold')
    parser.add_argument(
            '--keep_ratio', type=float, default=1, help='Keep ratio')
    parser.add_argument(
        '--qa_model_type', type=str, default='allenai/unifiedqa-t5-large', help='model used for QA')
    parser.add_argument(
        '--qa_model_ckpt', type=str, default=None, help='model ckpt used for QA')
    parser.add_argument(
        '--max_input_len', type=int, default=256, help='max length of the input prompt')
    parser.add_argument(
        '--reward_shape', type=int, default=4, help='refer to reward.py for implementation of each option')
    parser.add_argument('--dataset_paths', type=str, nargs='+', default=['data/auto/gen_data_v1_train.json','data/auto/gen_data_v1_dev.json'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
            '--kl_coef', type=float, default=0.2, help='coefficient for KL term in reward')
    parser.add_argument(
            '--ensembling', type=str, default='max', choices=['max', 'moe', 'poe', 'majority'], help='ensembling method for inference')
    parser.add_argument("--num_workers",type=int,default=None)
    parser.add_argument("--keys",type=str,nargs='*',default=['answers','knowledge','KQknowledge'])
    args = parser.parse_args()


    if args.num_workers:
        import multiprocessing
        def init():
            global reward 
            process_id = multiprocessing.current_process()._identity[0]-1
            reward = Reward(
                model_type=args.qa_model_type,
                model_ckpt=args.qa_model_ckpt,
                max_input_len=args.max_input_len,
                batch_size=args.batch_size,
                reward_shape=args.reward_shape,
                kl_coef=args.kl_coef,
                ensembling=args.ensembling,
                device=f"cuda:{process_id}",
            )
        def task(batch):
            return get_scores_for_batch(batch,reward)

        for path in args.dataset_paths:
            with open(path) as f:
                data = json.load(f)
            flat_data,flat_lengths = get_flat_data(data,args.keys)
            batches = batchify(flat_data,batch_size=args.batch_size)
            
            with multiprocessing.Pool(args.num_workers,initializer=init) as pool:
                result_chunks = list(tqdm(pool.imap(task,batches),total=len(batches)))
                flat_scores = []
                flat_answer_probs = []
                for result in result_chunks:
                    flat_scores.extend(result[0])
                    flat_answer_probs.extend(result[1])
            #update data
            start = 0
            for i in range(len(data)):
                data[i]['scores'] = flat_scores[start:start+flat_lengths[i]]
                data[i]['probs'] = flat_answer_probs[start:start+flat_lengths[i]]
                start += flat_lengths[i]
            
            # filtered_data = filter_data(data,keep_threshold=args.keep_threshold,keep_ratio=args.keep_ratio)
            
            with open(path+".scored",'w') as f:
                json.dump(data,f,indent=2,ensure_ascii=False)
            # with open(path+".filter",'w') as f:
            #     json.dump(filtered_data,f)
    else:
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            device="cuda",
        )

        for path in args.dataset_paths:
            with open(path) as f:
                data = json.load(f)
            data = get_scores(data,reward_model=reward,batch_size=args.batch_size)
            filtered_data = filter_data(data,keep_threshold=args.keep_threshold,keep_ratio=args.keep_ratio)
            with open(path+".scored",'w') as f:
                json.dump(data,f,indent=2,ensure_ascii=False)
            with open(path+".filter",'w') as f:
                json.dump(filtered_data,f,indent=2,ensure_ascii=False)
