import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
import argparse

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, AdamW, Trainer, TrainingArguments, BertConfig, RobertaForSequenceClassification, RobertaConfig, RobertaModel, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
from load_data import *
from bert_entity import *
from pororo import Pororo
from loss import LabelSmoothingLoss

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1) # bert, xlm-roberta-large

  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def func_eval(model, data_iter, device):
  with torch.no_grad():
    n_total, n_correct = 0, 0
    model.eval()  # evaluate (affects DropOut을 안하고, and BN은 학습되어있는 것을 사용)
    for batch in tqdm(data_iter):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

      logits = outputs[1]
      y_pred = torch.argmax(logits, axis=-1)
      n_correct += (y_pred==labels).sum().item()

      n_total += len(y_pred)

    val_accr = (n_correct / n_total)
    model.train()  # back to train mode
  return val_accr


def custom_trainer(model, device, RE_train_dataset, RE_valid_dataset, args):
  train_loader = DataLoader(RE_train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=args['num_workers'])
  valid_loader = DataLoader(RE_valid_dataset, batch_size=args['eval_batch_size'], shuffle=True, num_workers=args['num_workers'])

  optim = AdamW(model.parameters(), lr=args['lr'])
  loss_fn = LabelSmoothingLoss()

  model.train()

  EPOCHS, print_every = args['epochs'], 1

  for epoch in range(EPOCHS):
    loss_val_sum = 0

    for batch in tqdm(train_loader):
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

      loss = outputs[0]

      loss.backward()
      optim.step()
      loss_val_sum += loss
    loss_val_avg = loss_val_sum / len(train_loader)

    if ((epoch % print_every) == 0 or epoch == (EPOCHS - 1)):

      train_accr = func_eval(model, train_loader, device)
      valid_accr = func_eval(model, valid_loader, device)
      print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] valid_accr:[%.3f]."%(epoch,loss_val_avg,train_accr,valid_accr))


def train(args):

  # device setting
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load model and tokenizer
  if args['model_name'] == "xlm-roberta-large":
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    config.num_labels = args['num_labels']

    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)


  elif args['model_name'] == "roberta-base":
    MODEL_NAME = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = RobertaConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    config.num_labels = args['num_labels']

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)


  elif args['model_name'] == "bert-base-multilingual-cased":
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = BertConfig.from_pretrained(MODEL_NAME)
    config.num_labels = args['num_labels']

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)


  else:
    MODEL_NAME = args['model_name']
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)

  # if you use entity_token
  if args['entity_token']:
    special_tokens_dict = {'additional_special_tokens': ["#", "@", '₩', '^']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

  # load dataset
  dataset = load_data("/opt/ml/input/data/train/train.tsv")
  train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, random_state=args['random_seed'])
  train_label = train_dataset['label'].values
  valid_label = valid_dataset['label'].values

  # pororo ner
  ner = Pororo(task="ner", lang="ko")

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, ner, args)
  tokenized_valid = tokenized_dataset(valid_dataset, tokenizer, ner, args)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

  # update model setting

  model.to(device)

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

  print("use_trainer : ", args['use_trainer'])

  if args['use_trainer']:
      training_args = TrainingArguments(
        output_dir='./results',                                 # output directory
        save_total_limit=5,                                     # number of total save model.
        save_steps=500,                                         # model saving step.
        num_train_epochs=args['epochs'],                        # total number of training epochs
        learning_rate=args['lr'],                               # learning_rate
        per_device_train_batch_size=args['train_batch_size'],   # batch size per device during training
        per_device_eval_batch_size=args['eval_batch_size'],     # batch size for evaluation
        warmup_steps=args['warmup_steps'],                      # number of warmup steps for learning rate scheduler
        weight_decay=args['weight_decay'],                      # strength of weight decay
        logging_dir='./logs',                                   # directory for storing logs
        logging_steps=args['logging_steps'],                                      # log saving step.
        label_smoothing_factor=args['label_smoothing_factor'],
        evaluation_strategy='steps',                            # evaluation strategy to adopt during training
                                                                # `no`: No evaluation during training.
                                                                # `steps`: Evaluate every `eval_steps`.
                                                                # `epoch`: Evaluate every end of epoch.
        eval_steps = 100,                                       # evaluation step.
      )
      trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
      )

      # train model
      trainer.train()

  else:
      custom_trainer(model, device, RE_train_dataset, RE_valid_dataset, args)



def main():

  parser = argparse.ArgumentParser()

  # Data and model checkpoints directories
  parser.add_argument('--random_seed', type=int, default=42, help='random seed (default = 42)')
  parser.add_argument('--entity_token', type=bool, default=False, help='special entity token')
  parser.add_argument('--epochs', type=int, default=10, help='epochs (default = 10)')
  parser.add_argument('--lr', type=float, default=1e-5, help='learning rate(default = 1e-5)')
  parser.add_argument('--use_trainer', type=bool, default=True, help='use huggingface trainer (default = True)')
  parser.add_argument('--max_length', type=int, default=180, help='sequence length (default = 180)')
  parser.add_argument('--train_batch_size', type=int, default=32, help='train_batch_size (default = 16)')
  parser.add_argument('--eval_batch_size', type=int, default=32, help='eval_batch_size (default = 16)')
  parser.add_argument('--label_smoothing_factor', type=int, default=0.5, help='Percentage of label smoothing factor used (default = 0.5)')
  parser.add_argument('--logging_steps', type=int, default=100, help='logging steps (default = 100)')
  parser.add_argument('--warmup_steps', type=int, default=300, help='warmup_steps (default = 300)')
  parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay (default = 0.01)')
  parser.add_argument('--num_workers', type=int, default=4, help='CPU num_workers (default = 4)')
  parser.add_argument('--num_labels', type=int, default=42, help='Number of labels (default = 42)')

  # import arguments
  parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='model_name')

  args = vars(parser.parse_args())
  train(args)

if __name__ == '__main__':
  main()
