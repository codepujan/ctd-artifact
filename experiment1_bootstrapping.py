from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import gc,torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# huggingface hub model id
model_id = "philschmid/flan-t5-xxl-sharded-fp16"
cache_dir="./"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto",cache_dir=cache_dir)

import pandas as pd
df=pd.read_csv("hf://datasets/ppaudel/dataset_climate/dataset_climate_triplet.csv")
tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",cache_dir=cache_dir)


def compute_f1(precision,recall):
    return 2*(precision*recall)/(precision+recall)
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
#
from collections import defaultdict
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

import numpy as np

def false_negative_rate(gts, preds):
    gts = np.array(gts)
    preds = np.array(preds)
    
    total_refuting = np.sum(gts == 0)
    false_negatives = np.sum((gts == 0) & (preds == 1))
    
    return false_negatives / total_refuting if total_refuting > 0 else 0
def compute_stats(results,results_gt):
    gts=[]
    preds=[]
    for index in range(0,len(results)):
        gts.append(0 if results_gt[index]["gt"].lower()=="refute" else 1)
        if "no" in results[index].lower() or "refute" in results[index].lower() or "ref" in results[index].lower():
            preds.append(0)
        elif "yes" in results[index].lower() or "support" in results[index].lower() or "sup" in results[index].lower():
            preds.append(1)
        else:
            print("Output space doesn't fall in support or refute ?",ground_truth.lower())
    print("F1 score",f1_score(gts,preds,average='weighted'))



def infer_and_stat_without_marker(df):
    results_gt=[]
    prompts=[]
    idx=-1
    for index,row in df.iterrows():
      idx+=1
      if 1:
            text=row["text"].replace("'","")
            if 1:
                input_text='''
                        Classify if a statement supports or refutes the scientific fact: {0}.
                            Statement: {1}.
                            Response: 
                        '''
                #
                prompts.append(input_text.format(row['consensus'],row['text']))
                results_gt.append({"text":text,"gt":row['stance']})
    #Now infer
    prompts_chunks= list(chunk(prompts, 50))
    results=[]
    for prompts in tqdm(prompts_chunks):
        input_ids = tokenizer(prompts, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids.to("cuda")
        outputs = model.generate(input_ids=input_ids)
        results_batch=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        results.extend(results_batch)
        del outputs
        del results_batch
        torch.cuda.empty_cache()
        gc.collect()
    compute_stats(results,results_gt)

print("Running bootstrapped CTD without markers")
infer_and_stat_without_marker(df)

print("-----")
###
def infer_and_stat_without_consensus(df):
    results_gt=[]
    prompts=[]
    idx=-1
    for index,row in df.iterrows():
      idx+=1
      if 1:
            text=row["text"].replace("'","")
            if 1:
                input_text='''
                        Classify if a given statement supports or refutes a scientific consensus.
                        Statement: {0}.
                        Output: Supports.
                        Statement: {1}.
                        Output: Refutes.
                        Statement: {2}.
                        Output: 
                    '''
                prompts.append(input_text.format(row['consensus'],row['refuting'],row['text']))#Config with consensus instead of supporting performs better
                results_gt.append({"text":text,"gt":row['stance']})
    #Now infer
    prompts_chunks= list(chunk(prompts, 50))
    results=[]
    for prompts in tqdm(prompts_chunks):
        input_ids = tokenizer(prompts, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids.to("cuda")
        outputs = model.generate(input_ids=input_ids)
        results_batch=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        results.extend(results_batch)
        del outputs
        del results_batch
        torch.cuda.empty_cache()
        gc.collect()
    compute_stats(results,results_gt)
print("Running bootstrapped CTD on Climate Skepticisim without consensus")
infer_and_stat_without_consensus(df)

print("-----")
###
def infer_and_stat_full(df):
    results_gt=[]
    prompts=[]
    idx=-1
    for index,row in df.iterrows():
      idx+=1
      if 1:
            text=row["text"].replace("'","")
            if 1:
                input_text='''
                    Classify if a statement supports or refutes the scientific fact: {0}.
                        Statement: {1}.
                        Response: Supports.
                        Statement: {2}.
                        Response: Refutes.
                        Statement: {3}.
                        Response:
                    '''
                prompts.append(input_text.format(row['consensus'],row['supporting'],row['refuting'],row['text']))#Config with consensus instead of supporting performs better
                results_gt.append({"text":text,"gt":row['stance']})
    #Now infer
    prompts_chunks= list(chunk(prompts, 50))
    results=[]
    for prompts in tqdm(prompts_chunks):
        input_ids = tokenizer(prompts, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids.to("cuda")
        outputs = model.generate(input_ids=input_ids)
        results_batch=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        results.extend(results_batch)
        del outputs
        del results_batch
        torch.cuda.empty_cache()
        gc.collect()
    compute_stats(results,results_gt)
print("Running bootstrapped CTD on Climate Skepticisim with full config ...")
infer_and_stat_full(df)
