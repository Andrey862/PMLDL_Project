# Project Deliverable D1.4
I am doing this project as a part of my thesis

The topic is finding a relation between nested named entities in a [Russian Dataset](https://arxiv.org/pdf/2108.13112.pdf)

For example "Mayer of Moscow" we have two Named entity: "Mayer of Moscow" - PROFESSION and "Moscow" - CITY and a relation: "Mayer of Moscow" WORKS_IN  "Moscow" 

At this point I done several experiments with BERT-based models trying to classify the presence of a relation between nested entities 


# The log

## Binary: presence/abcence of a relation

Train a model to binary classification of presence/abcence of relation

"NEREL/dev" is used as test dataset

### Set 1 

#### Experimant 1 - BERT

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| BERT  | 20     | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=5, cooldown = 1, factor = 0.5)| 0.86 | 0.82|

Data example: 
"наследника [ британской ] короны"

#### Experimant 2 - BERT + context

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| BERT  | 20     | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=5, cooldown = 1, factor = 0.5)| 0.86 (the same) | 0.82 (the same)|

Context: 50 characters to the left and right of the word
Example: "лали совместное заявление к 75 - летию исторической { встречи [ американских ] и советских войск } в конце Второй мировой войны на реке Эльба в Герм"

#### Experimant 3 - BERT + entity tags/types

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| modyfied BERT | 20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=5, cooldown = 1, factor = 0.5)| 0.85 | 0.79 |

Model:
```python=
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            max_length = MAX_LENGTH # 100
            )
        self.dropout = nn.Dropout(0.1)
        self.emb = nn.Embedding(29, 10)
        self.fc = nn.Sequential(
            nn.Linear(768 + 10*2, 100),
            nn.LeakyReLU(True),
            nn.Linear(100, 1)
            )
        
    def forward(self, ent1, ent2, input_ids, token_type_ids, attention_mask):
        ent1 = self.emb(ent1)
        ent2 = self.emb(ent2)
        out = self.bert(input_ids = input_ids, token_type_ids= token_type_ids, attention_mask = attention_mask)[1]
        out = self.dropout(out)
        out = torch.cat((out, ent1, ent2), dim=1)
        out = self.fc(out)
        return out
```

#### Experimant 4 - BERT + entity tags/types + context

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| modyfied BERT (the same as in ex. 3) | 40 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.74 | 0.69 |

#### Experimant 5 - Simple statistic

| Model      | test F1 | test accuracy |
| ---------- | ------- | ------------- |
| Dictionary | 0.78    | 0.75          |


## Set 2

#### Experimant 1 - BERT on "A [SEP] AB" / "AB [SEP] A"

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| BERT | 20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.84 | 0.88 |

Главного управления МВД по **Москве** [SEP] Москве [SEP]


#### Experimant 2 - BERT on "A [SEP] AB" / "AB [SEP] A" and tags

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| modyfied BERT | 20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.84 | 0.88 |

```python=
from transformers import BertTokenizer, BertModel
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            max_length = MAX_LENGTH
            )
        self.dropout = nn.Dropout(0.1)
        self.emb = nn.Embedding(29, 10)
        self.fc = nn.Sequential(
            nn.Linear(768 + 10*2, 2)
            )
        
    def forward(self, ent1, ent2, input_ids, attention_mask, token_type_ids, labels):
        ent1 = self.emb(ent1)
        ent2 = self.emb(ent2)
        out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)[1]
        out = self.dropout(out)
        out = torch.cat((out, ent1, ent2), dim=1)
        out = self.fc(out)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out.view(-1, 2), labels.view(-1))
        return loss, out
```

#### Experimant 3 - BERT on "A [SEP] AB" / "AB [SEP] A" and tags

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| another modyfied BERT | 20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.86 | 0.89 |

```python=
from transformers import BertTokenizer, BertModel
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence", # Use the 12-layer BERT model, with an uncased vocab.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            max_length = MAX_LENGTH
            )
        self.dropout = nn.Dropout(0.1)
        self.emb = nn.Embedding(29, 10)
        self.fc = nn.Sequential(
            nn.Linear(768 + 10*2, 200),
            nn.LeakyReLU(True),
            nn.Linear(200, 200),
            nn.LeakyReLU(True),
            nn.Linear(200, 2)
            )
        
    def forward(self, ent1, ent2, input_ids, attention_mask, token_type_ids, labels):
        ent1 = self.emb(ent1)
        ent2 = self.emb(ent2)
        out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)[1]
        out = self.dropout(out)
        out = torch.cat((out, ent1, ent2), dim=1)
        out = self.fc(out)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out.view(-1, 2), labels.view(-1))
        return loss, out
```

#### Experimant 3B - BERT on "A [SEP] AB" and tags
I screwed up dataset in ex3:

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| another modyfied BERT | 20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.82 | 0.86 |

#### Experimant 4 - Mix dataset

Dataset with mix of nested and non-nested tags -> dataset of only nested tags

No significant impact

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| another modyfied BERT | 10+20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.82 | 0.86 |


<details>
  <summary>Misses</summary>
 

PROFESSION ---> ORGANIZATION : 25

ORGANIZATION ---> COUNTRY : 14

EVENT ---> LOCATION : 12

EVENT ---> PERSON : 10

ORGANIZATION ---> ORGANIZATION : 9

PROFESSION ---> COUNTRY : 8

ORGANIZATION ---> STATE_OR_PROVINCE : 8

ORGANIZATION ---> PERSON : 6

EVENT ---> COUNTRY : 5

PROFESSION ---> PROFESSION : 5

PROFESSION ---> STATE_OR_PROVINCE : 5

FACILITY ---> PERSON : 5

ORGANIZATION ---> RELIGION : 4

ORGANIZATION ---> PROFESSION : 4

EVENT ---> EVENT : 4

WORK_OF_ART ---> PERSON : 4

ORGANIZATION ---> CITY : 4

ORGANIZATION ---> IDEOLOGY : 4

EVENT ---> CITY : 3

ORGANIZATION ---> LOCATION : 3

EVENT ---> DISEASE : 3

LAW ---> LAW : 3

PENALTY ---> MONEY : 3

EVENT ---> STATE_OR_PROVINCE : 3

DISEASE ---> DISEASE : 3

PROFESSION ---> LOCATION : 2

LAW ---> COUNTRY : 2

PROFESSION ---> CITY : 2

EVENT ---> WORK_OF_ART : 2

FACILITY ---> ORGANIZATION : 2

WORK_OF_ART ---> COUNTRY : 2

LAW ---> ORGANIZATION : 1

PRODUCT ---> ORGANIZATION : 1

CRIME ---> LAW : 1

PROFESSION ---> EVENT : 1

PROFESSION ---> PERSON : 1

ORGANIZATION ---> DISTRICT : 1

PENALTY ---> DATE : 1

FACILITY ---> DISTRICT : 1

ORGANIZATION ---> NUMBER : 1

AWARD ---> EVENT : 1

EVENT ---> PROFESSION : 1

WORK_OF_ART ---> LOCATION : 1

AWARD ---> DATE : 1

FACILITY ---> STATE_OR_PROVINCE : 1

EVENT ---> ORGANIZATION : 1

ORGANIZATION ---> FACILITY : 1

PERSON ---> COUNTRY : 1

PERSON ---> FAMILY : 1

WORK_OF_ART ---> CITY : 1

AWARD ---> COUNTRY : 1

LOCATION ---> CITY : 1

STATE_OR_PROVINCE ---> STATE_OR_PROVINCE : 1

ORGANIZATION ---> NATIONALITY : 1

LAW ---> PERSON : 1

LOCATION ---> COUNTRY : 1


</details>

#### Experiment 5 - Train BERT to predict non-binary labels but test on binary ones

| Model | Epochs | Optimizer, LR                | Sceduler | test F1 | test accuracy |
| ----- | ------ | ---------------------------- | -------- | ------- | ------------- |
| another modyfied BERT | 10+20 | AdamW(lr = 5e-5, eps = 5e-8) | ReduceLROnPlateau(patience=3, cooldown = 1, factor = 0.5)| 0.85 | 0.88 |



<details>
  <summary>Misses</summary>

PROFESSION ---> ORGANIZATION : 25

ORGANIZATION ---> ORGANIZATION : 14

ORGANIZATION ---> STATE_OR_PROVINCE : 10

EVENT ---> LOCATION : 10

ORGANIZATION ---> COUNTRY : 9

EVENT ---> PERSON : 8

PROFESSION ---> COUNTRY : 7

FACILITY ---> PERSON : 7

EVENT ---> COUNTRY : 6

WORK_OF_ART ---> PERSON : 6

PROFESSION ---> STATE_OR_PROVINCE : 6

ORGANIZATION ---> PERSON : 5

LAW ---> LAW : 5

EVENT ---> EVENT : 5

DISEASE ---> DISEASE : 5

PROFESSION ---> PROFESSION : 5

EVENT ---> DISEASE : 4

ORGANIZATION ---> PROFESSION : 4

EVENT ---> STATE_OR_PROVINCE : 4

LAW ---> PERSON : 3

PENALTY ---> MONEY : 3

ORGANIZATION ---> CITY : 3

ORGANIZATION ---> LOCATION : 2

LAW ---> COUNTRY : 2

PROFESSION ---> LOCATION : 2

ORGANIZATION ---> DISTRICT : 2

EVENT ---> WORK_OF_ART : 2

EVENT ---> FACILITY : 2

WORK_OF_ART ---> COUNTRY : 2

ORGANIZATION ---> RELIGION : 2

ORGANIZATION ---> NATIONALITY : 2

LAW ---> ORGANIZATION : 1

EVENT ---> ORGANIZATION : 1

LAW ---> STATE_OR_PROVINCE : 1

PROFESSION ---> RELIGION : 1

CRIME ---> LAW : 1

PROFESSION ---> EVENT : 1

PROFESSION ---> PERSON : 1

PERSON ---> PERSON : 1

FACILITY ---> DISTRICT : 1

AWARD ---> EVENT : 1

EVENT ---> PROFESSION : 1

WORK_OF_ART ---> LOCATION : 1

ORGANIZATION ---> IDEOLOGY : 1

AWARD ---> DATE : 1

ORGANIZATION ---> FACILITY : 1

PERSON ---> COUNTRY : 1

PERSON ---> FAMILY : 1

EVENT ---> CITY : 1

WORK_OF_ART ---> CITY : 1

FACILITY ---> LOCATION : 1

FACILITY ---> ORGANIZATION : 1

LOCATION ---> CITY : 1

STATE_OR_PROVINCE ---> STATE_OR_PROVINCE : 1

LOCATION ---> COUNTRY : 1
</details>