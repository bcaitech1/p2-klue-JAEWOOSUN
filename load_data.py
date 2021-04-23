import pickle as pickle
import os
import pandas as pd
import torch

from tqdm.auto import tqdm


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])

    return item

  def __len__(self):
    return len(self.labels)



def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 'entity_01_idx0':dataset[3], \
                              'entity_01_idx1':dataset[4], 'entity_02':dataset[5], 'entity_02_idx0':dataset[6], \
                              'entity_02_idx1':dataset[7], 'label':label,})
  return out_dataset



# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset



def return_tag(tagging_list, is_first):
    tag = ''
    if len(tagging_list) != 1:
        tagging = [tag[1] for tag in tagging_list if tag[1] != 'O']
        if tagging:
            tag = ' '.join(list(set(tagging)))
        else:
            tag = 'o'

    else:
        tag = tagging_list[0][1]

    assert tag!='', 'tagging이 빔'

    if is_first:
        return ' ₩ ' + tag.lower() + ' ₩ '
    else:
        return ' ^ ' + tag.lower() + ' ^ '



def tokenized_dataset(dataset, tokenizer, ner, args):

  concat_entity = []
  sentence_list = []

  # if you use ner_entity_token
  if args['entity_token']:
      for sent, ent01, ent02, start1, end1, start2, end2 in tqdm(zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02'],\
              dataset['entity_01_idx0'], dataset['entity_01_idx1'], dataset['entity_02_idx0'], dataset['entity_02_idx1']), total=len(dataset['sentence'])):

          ner_01 = return_tag(ner(ent01), True)
          ner_02 = return_tag(ner(ent02), False)

          temp = '#'+ner_01 + ent01+' # ' + '[SEP]' + '@' +ner_02+ ent02 + ' @ '
          concat_entity.append(temp)

          start1, end1 = int(start1), int(end1)
          start2, end2 = int(start2), int(end2)

          if start1 < start2:
              sent = sent[:start1]+'#'+ner_01+sent[start1:end1+1]+' # '+sent[end1+1:start2]+\
                  '@'+ner_02+sent[start2:end2+1]+ ' @ '+sent[end2+1:]
          else:
              sent = sent[:start2] + '@' + ner_02 + sent[start2:end2 + 1] + ' @ ' + sent[end2 + 1:start1] + \
                     '#' + ner_01 + sent[start1:end1 + 1] + ' # ' + sent[end1 + 1:]

          sentence_list.append(sent)

  else:
      for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
          temp = e01 + '[SEP]' + e02
          concat_entity.append(temp)

      sentence_list = list(dataset['sentence'])

  # use tokenizer
  tokenized_sentences = tokenizer(
      concat_entity,
      sentence_list,
      return_tensors="pt",
      max_length=args['max_length'],
      padding='max_length',
      truncation='longest_first',
      add_special_tokens=True,
      )
  return tokenized_sentences
