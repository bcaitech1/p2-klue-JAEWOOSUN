Pstage_02_KLUE_Relation_extraction
=====


## 1) 목표

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.

총 42개의 Label로 분류합니다.

* input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.

entity 1: 썬 마이크로시스템즈

entity 2: 오라클

relation: 단체:별칭
```


* output: relation 42개 classes 중 1개의 class를 예측한 값입니다.


#
## 2) 사용법   
      
### [Dependencies]


* torch==1.6.0

* pandas==1.1.5

* scikit-learn~=0.24.1
  
* transformers==4.2.0

### [Install Requirements]   

    pip install -r requirements.txt

### [Training]

* Terminal 실행   
  
  parser를 통해 실행할 수 있습니다.

```
python train.py --epochs 30 --batch_size 16 --model xlm-roberta-large
```

### [Inference]

* Terminal 실행   
  
  parser를 통해 실행할 수 있습니다.

  --model_dir을 통해 원하는 checkpoint 위치의 model weight를 받아올 수 있습니다.

```
python Inference.py  --model_dir=./results/checkpoint-500 --epochs 30 --batch_size 16 --model xlm-roberta-large 
```
#
## 3) 파일 설명

* train.py   
  
  - huggingface trainer, model의 실행, eval Accuracy 출력

* load_data.py
  
  - Dataset 구현, tokenizer 구현

* Inference.py
  
  - test data 예측


#
## 4) 주요 코드 설명

## [train.py]   

- compute_metrics 함수  
  
  huggingface Trainer에서 validation set의 accuracy, loss를 계산해줍니다.

  - parameter
    - pred : model output tensor
  - returns
    - accuracy, f1 등을 담은 dictionary 

  (19줄)

```
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1) # bert, xlm-roberta-large

  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
```

- 모델 불러오기
  
  원하는 Model과 tokenizer를 불러올 수 있습니다.

  (91줄)

```
MODEL_NAME = "xlm-roberta-large"
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    config.num_labels = args['num_labels']

    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
```

- pororo 라이브러리 사용
  
  Kakao brain에서 발표한 pororo 라이브러리를 불러옵니다.

  pororo 라이브러리는 load_data.py의 tokenizer 전처리에 사용됩니다.

  - parameter
    - task : ner task를 수행하기 때문에 "ner"을 넣어줍니다.
    - lang : "ko"

  (137줄)

```
ner = Pororo(task="ner", lang="ko")
```

- training_args
  
  huggingface의 trainer에서 여러가지 parameter 설정이 가능합니다.

  - parameter 
    - output_dir : model checkpoint를 저장할 path
    - save_total_limit : 저장할 checkpoint 최대 개수
    - sava_steps : 저장할 checkpoint step
    - num-train_epochs : epochs
    - learning_rate : learning rate
    - per_device_train_batch_size : train batch size
    - per_device_eval_batch_size : valid batch size
    - warmup_steps : 처음 learning rate를 천천히 올릴 step 수
    - weight_decay : weight decay
    - logging dir : log를 저장할 위치
    - logging_steps : log를 저장할 step 수
    - label_smoothing_factor : label smoothing을 적용할 비율
    - evaluation_strategy : validation set에 적용할 전략
    - eval_steps : validation set을 eval할 적용할 step

  (156줄)

```
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
```

## [load_data.py]

- RE_Dataset
  
  tokenized된 input들을 다시 한 번 dataset으로 만드는 과정입니다.

  (10줄)

```
RE_train_dataset = RE_Dataset(tokenized_train, train_label)
```

- return_tag

  - parameter
     - tagging_list : tuple 형태의 (name, 객체명)
     - is_first : 첫번째 문장인지
   - returns
     - ner이 적용된 tag

    Pororo ner을 통과한 문장 (tagging_list)에서 하나를 선택주는 것이 아닌, 'o'를 제외하고 하나의 문자열로 합쳐서 entity의 개체명으로 사용하는 방법입니다.

    (54줄)

```
ner_01 = return_tag(ner(ent01), True)
```

- tokenized_dataset

   - parameter
     - dataset
     - tokenizer
     - ner : pororo ner
     - args : args parser
   - returns
     - tokenized된 object
  
    sentence가 tokenizer에 들어가기 전 전처리를 하고, tokenizing하는 함수입니다.
    
    (75줄)

```
tokenized_train = tokenized_dataset(train_dataset, tokenizer, ner, args)
```

## [inference.py]

- inference 함수
  
  * parameter 
    * model : inference를 수행할 Sequence classification 모델
    * dataset : test dataset
    * device : GPU or CPU

  * returns
    * 예측 label값 (numpy array)

  (12줄)

```
pred_answer = inference(model, test_dataset, device)
```