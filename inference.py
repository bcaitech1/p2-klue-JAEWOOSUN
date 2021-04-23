from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

from pororo import Pororo

def inference(model, tokenized_sent, device):

  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []

  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values

  # pororo ner
  ner = Pororo(task="ner", lang="ko")

  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, ner)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


  # load tokenizer
  TOK_NAME = "xlm-roberta-large"
  tokenizer = XLMRobertaTokenizer.from_pretrained(TOK_NAME)

  special_tokens_dict = {'additional_special_tokens': ["#", "@", '₩', '^']}
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

  # load my model
  MODEL_NAME = args.model_dir # model dir.
  model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME)
  model.resize_token_embeddings(len(tokenizer))
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device)

  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv('./prediction/submission.csv', index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
  args = parser.parse_args()
  print(args)
  main(args)
  
