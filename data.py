import csv,json,os,re

PERSON_NAMES = ['Alex', 'Alice', 'Aspen', 'Bali', 'Benjamin', 'Cameron', 'Chris', 'Cody', 'Dana', 'Davis', 'Emory', 'Flynn', 'Gale', 'Jamie', 'Jesse', 
'Kai', 'Kendall', 'Kyle', 'Lee', 'Logan', 'Micheal', 'Morgan', 'Nico', 'Parker', 'Patrick', 'Quinn', 'Ray', 'Robin', 'Rowan', 'Rudy', 'Sarah', 'Skylar', 'Susan', 
'Taylor', 'Tracy', 'Wilson', 'Wynne']
#这一系列载入数据都是载入平坦的三元组数据
def load_atomic_dataset(path):
    data={}
    with open(path) as source_file:
        source_reader = csv.reader(source_file)
        # Read first line to get column name
        source_line = next(source_reader)
        event_colname = source_line[0]
        categories_colname = source_line[1:10]
        prefix_colname = source_line[10]
        split_colname = source_line[11]
        for source_line in source_reader:
            # get every column
            event = source_line[0]
            annotationss = [
                json.loads(raw_anns) for raw_anns in source_line[1:10]]
            event_prefix = source_line[10]
            event_split = source_line[11]
            if event_split not in data:
                data[event_split] = []
            for category,annotations in zip(categories_colname,annotationss):
                for annotation in annotations:
                    #can do some normalize here, e.g., uncase, replace ___, normalize person xyz. 
                    d = {"h":event,"r":category,"t":annotation}
                    data[event_split].append(d)
    return data
def load_atomic2020_dataset(path):
    splits = ['train','dev','test']
    data = {}
    for split in splits:
        data[split] = []
        with open(os.path.join(path,f"{split}.tsv")) as f:
            source_reader = csv.reader(f,dialect='excel-tab')
            for line in source_reader:
                # line = line.rstrip('\n').split('\t')
                #can do some normalize here
                if line[0] and line[1] and line[2]: 
                    h,r,t = line
                    #normalization, may influence the results
                    t = normalize_tail(t) #
                    if t:
                        d = {"h":h,"r":r,"t":t}
                        data[split].append(d)
    return data
_dataset_loaders = {
    "atomic":load_atomic_dataset,
    "atomic2020":load_atomic2020_dataset,
}
def load_dataset(dataset_name,dataset_path):
    return _dataset_loaders[dataset_name](dataset_path)

_norm_symbols = {'\t':' ',
 '`':"'",
 '´':"'",
 'é':'e',
 'ê':'e',
 '—':'_',
 '’':"'"
}
_strip_symbols = ''.join({
 '\t',
 ' ',
 "'",
 ',',
 '.',
 '/',
 ':',
 '=',
 '>',
 '\\',
 '`',
 '´',
 '—',
 '’',
})
subline_regex = re.compile(r'_+')
multispace_regex = re.compile(r' +')
personx_regex = re.compile(r'\b[Pp]erson[Xx]\b|\b[Pp]erson [Xx]\b|\b[Xx]\b')
persony_regex = re.compile(r'\b[Pp]erson[Yy]\b|\b[Pp]erson [Yy]\b|\b[Yy]\b')
personz_regex = re.compile(r'\b[Pp]erson[Zz]\b|\b[Pp]erson [Zz]\b|\b[Zz]\b')
def normalize_tail(tail):
    tail = tail.strip(_strip_symbols)
    tail = ''.join([_norm_symbols[w] if w in _norm_symbols else w for w in tail])
    tail = subline_regex.sub('___',tail) 
    tail = multispace_regex.sub(' ',tail)
    #don't do case mapping
    # if tail in case_mapping:
    #     tail = case_mapping[tail]
    tail = personx_regex.sub("PersonX",tail)
    tail = persony_regex.sub("PersonY",tail)
    tail = personz_regex.sub("PersonZ",tail)
    tail = 'none' if tail.lower() == 'none' else tail #合并所有none
    return tail


#

from torch.utils.data import Dataset
import random



class SynQADatasetPlus(Dataset):
    def __init__(self, split, dataset_paths,dataset_name='CKG',random_naming=None):
        self.split = split
        self.dataset_paths = dataset_paths
        self.dataset_name = dataset_name
        self.instances = self.load_datasets(random_naming)

        # if split == 'train':
            # random.shuffle(self.instances)
        random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def give_random_name(self,d):
        names = random.sample(PERSON_NAMES,3)
        if "PersonX" in d['query']: 
            d['query'] = d['query'].replace("PersonX",names[0])
            d['choices'] = [choice.replace("PersonX",names[0]) for choice in d['choices']]
        if "PersonY" in d['query']: 
            d['query'] = d['query'].replace("PersonY",names[1])
            d['choices'] = [choice.replace("PersonY",names[1]) for choice in d['choices']]
        if "PersonZ" in d['query']: 
            d['query'] = d['query'].replace("PersonZ",names[2])
            d['choices'] = [choice.replace("PersonZ",names[2]) for choice in d['choices']]
        return d

    def load_datasets(self,random_naming):
    
        instances = []
        for dataset_path in self.dataset_paths:
            skipped = 0
            # datapath_by_split = datapath_by_task_and_split[task]
            # datapath = datapath_by_split[self.split if self.split in datapath_by_split else 'default']
            with open(os.path.join(dataset_path, f'{self.split}.json')) as f:
                dataset = json.load(f)
                for line in dataset:
                    if random_naming is not None and random.random()<random_naming:
                        line = self.give_random_name(line)
                    line['choices'] = [choice[4:] for choice in line['choices']]
                    instances.append({
                        'task': self.dataset_name,
                        'question': line['query'].replace("\n",' \\n '),
                        'choices': line['choices'],
                        'answer': line['choices'][line['answer_id']],
                        'answer_ix': line['answer_id'],
                    })
            print(f'Loaded dataset for `` split {self.split}, skipped {skipped} instances')
        return instances

    # Make a collate function to fix dataloader weird list batching
    @staticmethod
    def collate_fn(batch):
        batched_dict = {}
        all_keys = batch[0].keys()
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in batch:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        return batched_dict

class SynQADataset(Dataset):
    def __init__(self, split, dataset_paths,random_naming=None):
        self.split = split
        self.dataset_paths = dataset_paths

        self.instances = self.load_datasets(random_naming)

        # if split == 'train':
            # random.shuffle(self.instances)
        random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def give_random_name(self,d):
        names = random.sample(PERSON_NAMES,3)
        if "PersonX" in d['query']: 
            d['query'] = d['query'].replace("PersonX",names[0])
            d['choices'] = [choice.replace("PersonX",names[0]) for choice in d['choices']]
        if "PersonY" in d['query']: 
            d['query'] = d['query'].replace("PersonY",names[1])
            d['choices'] = [choice.replace("PersonY",names[1]) for choice in d['choices']]
        if "PersonZ" in d['query']:
            d['query'] = d['query'].replace("PersonZ",names[2])
            d['choices'] = [choice.replace("PersonZ",names[2]) for choice in d['choices']]
        return d

    def load_datasets(self,random_naming):
    
        instances = []
        for dataset_path in self.dataset_paths:
            skipped = 0
            # datapath_by_split = datapath_by_task_and_split[task]
            # datapath = datapath_by_split[self.split if self.split in datapath_by_split else 'default']
            with open(os.path.join(dataset_path, f'{self.split}.json')) as f:
                dataset = json.load(f)
                for line in dataset:
                    if random_naming is not None and random.random()<random_naming:
                        line = self.give_random_name(line)
                    line['choices'] = [choice[4:] for choice in line['choices']]
                    instances.append({
                        'task': "CKG",
                        'question': line['query'].replace("\n",' \\n '),
                        'choices': line['choices'],
                        'answer': line['choices'][line['answer_id']],
                        'answer_ix': line['answer_id'],
                    })
            print(f'Loaded dataset for `` split {self.split}, skipped {skipped} instances')
        return instances

    # Make a collate function to fix dataloader weird list batching
    @staticmethod
    def collate_fn(batch):
        batched_dict = {}
        all_keys = batch[0].keys()
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in batch:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        return batched_dict



datapath_by_task_and_split = {
    'obqa': {'default': 'uqa/openbookqa'},
    'arc_e': {'default': 'uqa/arc_easy'},
    'arc_h': {'default': 'uqa/arc_hard'},
    'ai2sci_e': {'default': 'uqa/ai2_science_elementary'},
    'ai2sci_m': {'default': 'uqa/ai2_science_middle'},
    'csqa': {'default': 'uqa/commonsenseqa'},
    'qasc': {'default': 'uqa/qasc'},
    'piqa': {'default': 'uqa/physical_iqa', 'test': 'uqa/physical_iqa_test'},
    'siqa': {'default': 'uqa/social_iqa', 'test': 'uqa/social_iqa_test'},
    'wg': {'default': 'uqa/winogrande_xl'},
    'numersense': {'default': 'numersense'},
    'riddlesense': {'default': 'riddlesense'},
    'quartz': {'default': 'quartz'},
    'hellaswag': {'default': 'hellaswag'},
    'csqa_ih':{'default':'uqa/commonsenseqa_inhouse'}
}

'''
tasks_by_split = {
    'train': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg'],
    'valid': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'],
    'test': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'],
}
'''

class QADataset(Dataset):
    def __init__(self, split, tasks):
        self.split = split
        self.tasks = tasks.split(',')

        self.instances = self.load_datasets()

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        def parse_choices(s):
            '''
            s: serialized_choices '(A) ... (B) ... (C) ...'
            '''
            choices = []
            key = 'A' if s.find('(A)') != -1 else 'a'
            while True:
                pos = s.find(f'({chr(ord(key) + 1)})')
                if pos == -1:
                    break
                choice = s[3:pos]
                s = s[pos:]
                choice = choice.strip(' ')
                choices.append(choice)
                key = chr(ord(key) + 1)
            choice = s[3:]
            choice = choice.strip(' ')
            choices.append(choice)
            return choices
    
        instances = []
        for task in self.tasks:
            skipped = 0
            datapath_by_split = datapath_by_task_and_split[task]
            datapath = datapath_by_split[self.split if self.split in datapath_by_split else 'default']
            with open(os.path.join('data/', datapath, f'{self.split}.tsv')) as f:
                for line in f:
                    try:
                        q, a = line.strip('\n').split('\t')
                        q = q.strip(' ')
                        a = a.strip(' ')
                        choices = parse_choices(q.split('\\n')[1].strip(' '))
                        if a == '-' or a == '':
                            answer_ix = 0
                        else:
                            answer_ix = choices.index(a)
                    except Exception as e:
                        skipped += 1
                        continue
                    instances.append({
                        'task': task,
                        'question': q,
                        'choices': choices,
                        'answer': a,
                        'answer_ix': answer_ix,
                    })
            print(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        return instances

    # Make a collate function to fix dataloader weird list batching
    @staticmethod
    def collate_fn(batch):
        batched_dict = {}
        all_keys = batch[0].keys()
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in batch:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        return batched_dict



provided_k_datapath = {
    "ecqa_posneg":{
        "task":"csqa",
        "dev":"data/knowledge/human/csqa/dev_posneg.json"
    },
    "ecqa_expl":{
        "task":"csqa",
        "dev":"data/knowledge/human/csqa/dev_expl.json"
    },
    "qasc_fact12":{
        "task":"qasc",
        "dev":"data/knowledge/human/qasc/qasc_fact12.json"
    },

    "gpt3.5turbo_1":{
        "dev":"data/auto/gen_realq_1.json"
    }
}

class QADataset_WithK(Dataset):
    def __init__(self, split, dataset_names):
        self.split = split
        self.dataset_names = dataset_names.split(',')
        self.instances = self.load_datasets()

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):    
        instances = []
        for dataset_name in self.dataset_names:
            datapath = provided_k_datapath[dataset_name][self.split]
            # task = provided_k_datapath[dataset_name]['task']
            with open(datapath) as f:
                for d in json.load(f):
                    if 'task' not in d:
                        d['task']=provided_k_datapath[dataset_name]['task']
                    instances.append(d)
        return instances

    # Make a collate function to fix dataloader weird list batching
    @staticmethod
    def collate_fn(batch):
        batched_dict = {}
        all_keys = batch[0].keys()
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in batch:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        return batched_dict
