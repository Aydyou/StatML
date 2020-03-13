#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import torch
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

prefix = 'data/'
#train_df = pd.read_csv(prefix + 'train.csv', header=None)


test_df = pd.read_csv(prefix + 'test.csv', header=None)
#train_df[0] = (train_df[0] == 2).astype(int)
#test_df[0] = (test_df[0] == 2).astype(int)
#train_df.head()


# In[2]:


test_df.head()


# In[3]:


#train_df = pd.DataFrame({
#    'id':range(len(train_df)),
#    'label':train_df[0],
#    'alpha':['a']*train_df.shape[0],
#    'text': train_df[1].replace(r'\n', ' ', regex=True)
#})

#train_df.head()


# In[4]:


dev_df = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

dev_df.head()


# In[5]:


#train_df.to_csv('data/train.tsv', sep='\t', index=False, header=False)
dev_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)


# In[6]:


import io
import sys

sys.stdin.reconfigure(encoding='utf-8')
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
import locale
def getpreferredencoding(do_setlocale = True):
    return "utf-8"
locale.getpreferredencoding = getpreferredencoding


# In[ ]:





# In[7]:


#from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange
from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule
import utils

from utils import (convert_examples_to_features,
                        output_modes, processors)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[8]:


args = {
    'data_dir': 'data/',
    'model_type':  'xlnet',
    'model_name': 'xlnet-base-cased',
    'task_name': 'binary',
    'output_dir': 'outputs/',
    'cache_dir': 'cache/',
    'do_train': False,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 8,
    'eval_batch_size': 8,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 2000,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': False,
    'reprocess_input_data': True,
    'notes': 'Using Yelp Reviews dataset'
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[9]:


#args


# In[10]:


with open('args.json', 'w') as f:
    json.dump(args, f)


# In[11]:


if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))


# In[12]:


MODEL_CLASSES = {   
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]


# In[13]:


config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])


# In[14]:


model = model_class.from_pretrained(args['model_name'])


# In[15]:


model.to(device);


# In[16]:


task = args['task_name']

processor = processors[task]()
label_list = processor.get_labels()
num_labels = len(label_list)


# In[17]:


def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        
        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


# In[18]:


def train(train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)


    return global_step, tr_loss / global_step


# In[19]:


from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_mismatched(labels, preds)

def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    #results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)
    #results.update(result)

    #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        #for key in sorted(result.keys()):
        #    logger.info("  %s = %s", key, str(result[key]))
        #    writer.write("%s = %s\n" % (key, str(result[key])))

    return wrong


# In[20]:


if args['do_train']:
    train_dataset = load_and_cache_examples(task, tokenizer)
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


# In[21]:


if args['do_train']:
    if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args['output_dir'])
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))


# In[22]:


#results = {}
#if args['do_eval']:
checkpoints = [args['output_dir']]
#    if args['eval_all_checkpoints']:
checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        #logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #logger.info("Evaluate the following checkpoints: %s", checkpoints)
for checkpoint in checkpoints:
    global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    model = model_class.from_pretrained(checkpoint)
    model.to(device)
    
evaluate(model, tokenizer, prefix=global_step)


# In[25]:


from flask import Flask, request, render_template
import requests
import torch
from translator import de2en
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, BertForQuestionAnswering
tokenizer2=BertTokenizer.from_pretrained('bert-base-uncased')
BertNSP=BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model3 = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
url = 'http://localhost:3000/predict'

app = Flask(__name__, template_folder="templates")

# Load the model

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = de2en.translate(request.form.get('text1'))
        text2= de2en.translate(request.form.get('text2'))
        text3= de2en.translate(request.form.get('text3'))
        text2_toks = ["[CLS]"] + tokenizer2.tokenize(text2) + ["[SEP]"]
        text3_toks = tokenizer2.tokenize(text3) + ["[SEP]"]
        text=text2_toks+text3_toks

        indexed_tokens = tokenizer2.convert_tokens_to_ids(text2_toks + text3_toks)
        segments_ids = [0]*len(text2_toks) + [1]*len(text3_toks)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        BertNSP.eval()
        prediction = BertNSP(tokens_tensor, token_type_ids=segments_tensors)
        prediction=prediction[0] # tuple to tensor
        softmax = torch.nn.Softmax(dim=1)
        prediction_sm = softmax(prediction)
        #Question Answering
        question = de2en.translate(request.form.get('text4'))
        phrase = de2en.translate(request.form.get('text5'))
        question_toks = ["[CLS]"] + tokenizer2.tokenize(question) + ["[SEP]"]
        phrase_toks = tokenizer2.tokenize(phrase) + ["[SEP]"]
        input_ids=tokenizer2.convert_tokens_to_ids(question_toks+phrase_toks)
        token_type_ids = [0]*len(question_toks) + [1]*len(phrase_toks)
        start_scores, end_scores = model3(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer2.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        
        #Question Answering for the case the sentiment is negative
        question2 = 'What was bad?'
        phrase2 = data
        question_toks2 = ["[CLS]"] + tokenizer2.tokenize(question2) + ["[SEP]"]
        phrase_toks2 = tokenizer2.tokenize(phrase2) + ["[SEP]"]
        input_ids2=tokenizer2.convert_tokens_to_ids(question_toks2+phrase_toks2)
        token_type_ids2 = [0]*len(question_toks2) + [1]*len(phrase_toks2)
        start_scores2, end_scores2 = model3(torch.tensor([input_ids2]), token_type_ids=torch.tensor([token_type_ids2]))
        all_tokens2 = tokenizer2.convert_ids_to_tokens(input_ids2)
        answer2 = ' '.join(all_tokens2[torch.argmax(start_scores2) : torch.argmax(end_scores2)+1])

        # Make prediction
        with open(prefix+'test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["1", str(data)])
        test_df = pd.read_csv(prefix + 'test.csv', header=None)
        dev_df = pd.DataFrame({
            'id':range(len(test_df)),
            'label':test_df[0],
            'alpha':['a']*test_df.shape[0],
            'text': test_df[1].replace(r'\n', ' ', regex=True)
        })
        
        dev_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
        correct=evaluate(model, tokenizer, prefix=global_step)
        if len(correct)==0:
            return render_template('demo.html', sentiment='positive', sentencepred='The probability is '+ str((prediction_sm.data[0])[0])[7:-1], questionanswer=answer)
        else:
            return render_template('demo.html', sentiment='negative. '+ 'The reason is: '+answer2, sentencepred='The probability is '+ str((prediction_sm.data[0])[0])[7:-1], questionanswer=answer)
            
    return render_template('demo.html', sentiment='', sentencepred='', questionanswer='')
        
        
    
if __name__ == '__main__':
    app.run(port=3000, debug=False)


# In[7]:





# In[8]:





# In[ ]:




