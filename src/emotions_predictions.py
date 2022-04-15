import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer, XLMRobertaModel, \
    XLMRobertaTokenizer
from torch.autograd import Variable

# Setting up the device for GPU usage

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print('device', device)

torch.manual_seed(0)
np.random.seed(0)
if device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset_name = 'multiclass'

transformer_model = 'xlm-roberta-base'
MAX_LEN = 512
TRAIN_BATCH_SIZE = VALID_BATCH_SIZE = 12
accumulation_steps = 1  # 64/TRAIN_BATCH_SIZE
EPOCHS = 25
LEARNING_RATE = 5e-06
embedding_dim = 1024 if 'large' in transformer_model else 768
hidden_dim = 1024
num_layers = 1
dropout = 0.5
freeze_transforemer = False
linear_version = False
loader_class_weighting = True
loss_class_weighting = True
custom_loss_f1 = False
eps = 1e-10
MAX_VALID_SLOW_DOWN = 15
WEIGHT_DECAY = 0  # 1e-2
OPTIMIZE_METRIC = 'f1_micro'
ONLY_PREDICTION = True

Tokenizer = DistilBertTokenizer if 'distil' in transformer_model else (
    XLMRobertaTokenizer if 'xlm-roberta' in transformer_model else BertTokenizer)
Model = DistilBertModel if 'distil' in transformer_model else (
    XLMRobertaModel if 'xlm-roberta' in transformer_model else BertModel)

# from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
#
# MODEL_TYPE = 'xlm-roberta-base'
#
# tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)

tokenizer = Tokenizer.from_pretrained(transformer_model)


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.sentence[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long),
            'targets_ids': self.data.id[index],
        }

    def __len__(self):
        return self.len


def valid(epoch, model, testing_loader, draw_confusion_matrix=True, visualize_wandb_cm=False):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    cnt = 0
    confusion_matrix_plt = None
    with torch.no_grad():
        y_test_values = []
        y_predict = []
        targets_ids = []
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            targets_cat = to_categorical(data['targets'], output_dim).to(device)
            targets_ids.append(np.array(data['targets_ids']))
            outputs = model(ids, mask)  # .squeeze()
            # if len(targets)==1:
            #    outputs.unsqueeze_(0)
            # print(outputs)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            # print("big val\n",big_idx)
            # print("targets\n",targets)
            y_test_values.append(targets.cpu())
            y_predict.append(big_idx.cpu())

            n_correct += (big_idx == targets).sum().item()
            cnt += 1
        y_test_values = np.concatenate(y_test_values).ravel().tolist()
        y_predict = np.concatenate(y_predict).ravel().tolist()
        y_ids = np.concatenate(targets_ids).ravel().tolist()
        accuracy = accuracy_score(y_test_values, y_predict)
        average = 'macro' if dataset_name == 'multiclass' else 'binary'
        precision_macro = precision_score(y_test_values, y_predict, average=average)
        recall_macro = recall_score(y_test_values, y_predict, average=average)
        f1_macro = f1_score(y_test_values, y_predict, average=average)
        average = 'micro' if dataset_name == 'multiclass' else 'binary'
        precision_micro = precision_score(y_test_values, y_predict, average=average)
        recall_micro = recall_score(y_test_values, y_predict, average=average)
        f1_micro = f1_score(y_test_values, y_predict, average=average)

        print('accuracy', accuracy)
        print('precision_macro', precision_macro)
        print('recall_macro', recall_macro)
        print('f1_macro', f1_macro)

        print('precision_micro', precision_micro)
        print('recall_micro', recall_micro)
        print('f1_micro', f1_micro)

        # print('y_test_values',y_test_values)
        # print('y_pred',y_predict)
        # y_true=to_categorical(np.array(y_test_values),num_classes=output_dim)
        # y_pred=to_categorical(np.array(y_predict),num_classes=output_dim)

        # print('f1_custom',torch.eval(f1_custom_score(y_pred,y_true).data[0])
        if draw_confusion_matrix:
            cm = confusion_matrix(y_test_values, y_predict)
            ax = sns.heatmap(cm, annot=True, fmt="d", xticklabels=ordered_labels, yticklabels=ordered_labels)
            confusion_matrix_plt = cm
            plt.show()
    obj = {
        'epoch': epoch,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'predict_values': y_predict,
        'actual_values': y_test_values,
        'targets_ids': y_ids
    }
    return obj[OPTIMIZE_METRIC], obj
