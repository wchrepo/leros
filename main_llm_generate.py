#采用平直脚本编写，之后再考虑统一模型
#%%
import os
import sys,json
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
os.chdir(root_dir)

import qak_templates

os.environ['http_proxy'] = 'http://127.0.0.1:56666'
os.environ['https_proxy'] = 'http://127.0.0.1:56666'
import openai


openai.api_key = ""

knowledge_answer_examples = [
"""Knowledge: Batteries can produce steady electrical current. Wires, resistors or light bulbs cannot produce electrical current.
Answer: (B) a battery
""",
"""Knowledge: Rubbing hands produces heat because of friction. Dry tree surfaces has larger friction. 
Answer: (A) dry palms
""",
"""Knowledge: Electronic maps are the modern version of paper atlas.
Answer: (D) atlas
""",
"""Knowledge: The man in the blue shirt is getting haircut. One should sit still when getting a haircut.
Answer: (C) sits on the chair next to the sink.
""",
"""Knowledge: Limbs include arms and legs. Human has two arms and two legs.
Answer: (E) four
""",
"""Knowledge: Natural light of the sun provides energy for photosynthesis.
Answer: (D) plants sprouting, blooming and wilting
""",
"""Knowledge: A hand saw is used for making cuts. A hand drill is used for making holes.
Answer: (A) Use a hand saw to cut the handles.
""",
"""Knowledge: The stomach is part of the digestive system. Digestive system can break down food.
Answer: (C) breaks food into nutrients
""",
"""Knowledge: Lower pH means more acid.
Answer: (B) decreases
""",
"""Knowledge: In autumn, the days get short and the nights get long.
Answer: (E) leaves
""",
"""Knowledge: A messy room likely contains dirty clothes. Other options are irrelevant to cleaning.
Answer: (C) Pick up the dirty clothes
""",
"""Knowledge: There are more trees in the forests than in the fields.
Answer: (B) forests
"""
]

question_examples = [
"""An Italian scientist named Alessandro Volta invented the Voltaic pile in 1800. It was able to produce a steady electrical current. Based on this description, what is the modern equivalent of the Voltaic pile?
(A) a wire (B) a battery (C) a resistor (D) a light bulb
""",
"""George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
(A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion
""",
"""Google Maps and other highway and street GPS services have replaced what?
(A) united states (B) mexico (C) countryside (D) atlas (E) oceans
""",
"""The man in the center is demonstrating a hairstyle on the person wearing the blue shirt. the man in the blue shirt
(A) is standing on the sponge cutting the hair of the person wearing the blue shirt. (B) is doing the hairstyle with his hand and the hairspray. (C) sits on the chair next to the sink. (D) is being shown eye to eye.
""",
"""a typical human being has <mask> limbs.
(A) no (B) one (C) two (D) three (E) four (F) five (G) six (H) seven (I) eight (J) nine (K) ten
""",
"""The sun is responsible for
(A) puppies learning new tricks (B) children growing up and getting old (C) flowers wilting in a vase (D) plants sprouting, blooming and wilting
""",
"""How can I cut the handles of metal cutlery?
(A) Use a hand saw to cut the handles. (B) Use a hand drill to cut the handles.
""",
"""The stomach does what in the body?
(A) decreases its bodily water (B) kills all germs (C) breaks food into nutrients (D) stores bile (E) heat is produced (F) extracts water from food (G) get chemical reactions started (H) cause people to become sick.
""",
"""If Mona is making lemonade and she decreases the acidity of it by adding more lemons, what happens to its closeness to pH 0?
(A) increases (B) decreases
""",
"""I am around when the days get short, and the nights long.  I am everywhere that you can see.  I scream undertoe when you walk over me.  What am I?
(A) hit (B) rake (C) branch (D) tract (E) leaves
""",
"""Quinn wanted to help me clean my room up because it was so messy. What will Quinn want to do next?
(A) Eat messy snacks (B) help out a friend (C) Pick up the dirty clothes
""",
"""The geese prefer to nest in the fields rather than the forests because in the _ predators are more hidden.
(A) fields (B) forests
"""
]

question_with_correct_examples = [
"""An Italian scientist named Alessandro Volta invented the Voltaic pile in 1800. It was able to produce a steady electrical current. Based on this description, what is the modern equivalent of the Voltaic pile?
(A) a wire (B) a battery {correct} (C) a resistor (D) a light bulb
""",
"""George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
(A) dry palms {correct} (B) wet palms (C) palms covered with oil (D) palms covered with lotion
""",
"""Google Maps and other highway and street GPS services have replaced what?
(A) united states (B) mexico (C) countryside (D) atlas {correct} (E) oceans
""",
"""The man in the center is demonstrating a hairstyle on the person wearing the blue shirt. the man in the blue shirt
(A) is standing on the sponge cutting the hair of the person wearing the blue shirt. (B) is doing the hairstyle with his hand and the hairspray. (C) sits on the chair next to the sink. {correct} (D) is being shown eye to eye.
""",
"""a typical human being has <mask> limbs.
(A) no (B) one (C) two (D) three (E) four {correct} (F) five (G) six (H) seven (I) eight (J) nine (K) ten
""",
"""The sun is responsible for
(A) puppies learning new tricks (B) children growing up and getting old (C) flowers wilting in a vase (D) plants sprouting, blooming and wilting {correct}
""",
"""How can I cut the handles of metal cutlery?
(A) Use a hand saw to cut the handles. {correct} (B) Use a hand drill to cut the handles.
""",
"""The stomach does what in the body?
(A) decreases its bodily water (B) kills all germs (C) breaks food into nutrients {correct} (D) stores bile (E) heat is produced (F) extracts water from food (G) get chemical reactions started (H) cause people to become sick.
""",
"""If Mona is making lemonade and she decreases the acidity of it by adding more lemons, what happens to its closeness to pH 0?
(A) increases (B) decreases {correct}
""",
"""I am around when the days get short, and the nights long.  I am everywhere that you can see.  I scream undertoe when you walk over me.  What am I?
(A) hit (B) rake (C) branch (D) tract (E) leaves {correct}
""",
"""Quinn wanted to help me clean my room up because it was so messy. What will Quinn want to do next?
(A) Eat messy snacks (B) help out a friend (C) Pick up the dirty clothes {correct}
""",
"""The geese prefer to nest in the fields rather than the forests because in the _ predators are more hidden.
(A) fields (B) forests {correct}
"""
]

import re
# kq_pattern = re.compile(
#     r"\bKnowledge: (.*?)\nRelated Question: (.*?)\n(.*?)(?:\n\n|$)"
# )
kq_pattern = re.compile(
    r"\b(Knowledge: (?P<knowledge>.*?)\nRelated Question: (?P<question>.*?)\n(?P<choices>.*?))\n\n"
)

ka_pattern = re.compile(
    r"(?:\d+\.\s+)((\bKnowledge: (?P<knowledge>.*?)\n)?Answer: (?P<answer>.*?)(?:\n|```|$))"
)

def get_prompt_given_question_v1_0(question_examples,ka_examples,given_questions,num_example=1,num_input_question=1,rand_example=False,rand_question=False):
   
    while True:
        if rand_example:
            selected_idx = random.sample(list(range(len(question_examples))),num_example)
        else:
            selected_idx = list(range(num_example))
        prompt_template = [
            {"role": "system", "content": "For given commonsense question, your task is to think about related commonsense knowledge and then answer the question."},
        ]

        example_text = "```\n"
        for i,ix in enumerate(selected_idx):
            example_text+=f"{i+1}. {question_examples[ix]}\n"
        example_text += "```"
        prompt_template.append({"role": "user", "content": (
            "Use your commonsense knowledge to choose the correct answer for some questions.  You response should be in this form\n"
            "```\n"
            "1. Knowledge: {knowledge}\n"
            "Answer: ({option}) {answer}\n"
            f"```\n"
            "If there is not proper option, you can give `Answer: None`.\n"
            f"Now answer the following questions:\n{example_text}"
            )})
        
        # kq_example_text = "```\n"
        ka_example_text = ""
        for i, ix in enumerate(selected_idx):
            ka_example_text+=f"{i+1}. {ka_examples[ix]}\n"
        # kq_example_text += "```"
        prompt_template.append({"role": "assistant", "content": f"{ka_example_text}"})

        questions = []
        given_question_text = "```\n"
        for i in range(num_input_question):
            if rand_question:
                pop_index=random.randint(0,len(given_questions)-1)
            else:
                pop_index=0
            question = given_questions.pop(pop_index)
            questions.append(question)
            given_question_text+=f"{i+1}. {question}\n"
        given_question_text += "```"
        prompt_template.append({"role": "user", "content": f"Now answer the following questions in the same format:\n{given_question_text}"})
        yield prompt_template,questions
#%%

num_example = 3
num_input_question=10
rand_example=True
rand_question = True
num_generate = 50000
num_response = 5
base_sleep = 10
max_retry = 3
max_workers = 20
path = "data/auto/auto_knowledge_answer_given_synquestion_v1_0"
source_path = "data/knowledge/SynQATOMIC2020/train.json"
def parse_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        for m in kq_pattern.finditer(content):
            correct = None
            choices = []
            for choice_content in re.split(r"\([A-Z]\)",m['choices']):
                choice_content = choice_content.strip().strip(".")
                if choice_content:
                    if choice_content.endswith("{correct}"):
                        correct = len(choices)
                        choice_content = choice_content.removesuffix("{correct}").strip()
                    choices.append(choice_content)
            # for i,m_choice in enumerate(choice_pattern.finditer(m['choices'])):
            #     choices.append(m_choice['choice_text'])
            #     if m_choice['correct']:
            #         correct = i
            if correct is None:
                print(m.group(0))
            else:
                data.append({"knowledge": m['knowledge'], "question": m['question'], "choices": choices, "correct": correct})
    return data

def parse_data_syn(file_path):
    with open(file_path) as f:
        raw = json.load(f)
    data = []
    for raw_d in raw:
        d = {}
        d['question'] = raw_d['query']
        data.append(d)
    return data

source_data = parse_data_syn(source_path)
question_inputs = [ d['question']+'\n' for d in source_data]
# question_inputs = [d['question']+"\n"+" ".join([f"({chr(ord('A')+i)}) {choice}" for i,choice in enumerate(d['choices'])])+"\n" for d in source_data]
# %%
import time,os,random,json,pickle
from tqdm.auto import tqdm
all_responses = []
if not os.path.exists(f"{path}.ka.txt"):
    f = open(f"{path}.ka.txt",'w')
    q_examples = question_examples.copy()
    ka_examples = knowledge_answer_examples.copy()
    input_questions = question_inputs
    start = 0
else:
    f = open(f"{path}.ka.txt",'a')
    with open(f"{path}.state.pkl",'rb') as statef:
        state_dict = pickle.load(statef)
        q_examples = state_dict['q_examples']
        ka_examples = state_dict['ka_examples']
        input_questions = state_dict['input_questions']
        start = state_dict['start']

        # iter_prompts_name = state_dict['iter_prompts_name']
    q_examples = question_examples.copy()
    ka_examples = enhanced_knowledge_answer_examples.copy() 

iter_prompts = get_prompt_given_question_v1_0(q_examples,ka_examples,input_questions,num_example,num_input_question,rand_example,rand_question)
#%%

def dump_state():
    with open(f"{path}.state.pkl",'wb') as statef:
        state_dict = {'q_examples': q_examples, 'ka_examples': ka_examples, 'input_questions': input_questions, 'start': i, 'iter_prompts_name': iter_prompts.__name__}
        pickle.dump(state_dict,statef)

import atexit
@atexit.register
def handle_exit():
    print("exiting gracefully")
    try:
        wait_cache_finish()
    finally:
        dump_state()
        print("save state before exit")
        f.close()
        with open(f"{path}.{int(time.time())}.metadata.json",'w') as metaf:
            json.dump(all_responses,metaf,indent=2,ensure_ascii=False)

import multiprocessing as mp

cache = []
def check_cache():
    global cache
    for i,c in enumerate(cache):
        if c.ready():
            try:
                (messages,questions,response,results,should_write)=c.get()
                all_responses.append((messages,response))
                # f.write(messages[-1]["content"].split("\n")[1]+"\n")
                if should_write:
                    for matches in results:
                        for question,result in zip(questions,matches):
                            if result.startswith("Answer:"):
                                result = "Knowledge:\n"+result
                            f.write(f"Question: {question}"+result)
                    f.flush()
            except Exception as e:
                tqdm.write(e)
                raise e
            finally:
                del cache[i]
                break

def wait_cache_finish():
    if len(cache)>0:
        check_cache()



def task(messages,questions):
    retry = 1
    while True:
        should_write=True
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                n=num_response,
                )
        except Exception as e:
            tqdm.write(str(e))
            time.sleep(random.randint(base_sleep+5,base_sleep+15))
            if retry>max_retry:
                raise e
            retry+=1
            continue
        results = []
        for r in response['choices']:
            matches = [match.group(1).strip()+'\n\n' for match in ka_pattern.finditer(r['message']['content'])]
            if len(matches) == len(questions):
                results.append(matches)
        # results = [match.group(0).strip()+'\n\n' for match in ka_pattern.finditer(response['choices'][0]['message']['content'])]
        # if len(results)!=len(questions):
        if len(results) == 0:

            if retry<max_retry:
                retry+=1
                continue
            else:
                should_write=False
            # should_write = False
        break
    return (messages,questions,response,results,should_write)

with mp.Pool(max_workers) as pool:
    for i in tqdm(range(start,num_generate)):
        while True:
            if len(cache)<max_workers:
                messages,questions = next(iter_prompts)
                cache.append(pool.apply_async(task,(messages,questions)))
                break
            check_cache()
            time.sleep(0.1)
        # dump_state()
        time.sleep(2)

