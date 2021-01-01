import pandas as pd
from torch.utils.data import Dataset

from config import *
from utils import *


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer):
        df = pd.read_csv(dataset_file_path, header=0, encoding='unicode_escape')
        self.items = df.values
        self.tokenizer = tokenizer
        self.data_text_ids = self.items[:, 0]
        self.data_texts = preprocessing(self.items[:, 1])
        self.data_encodings = self.tokenizer(self.data_texts.tolist(), truncation=True, padding=True, max_length=config.MAX_SEQ_LENGTH)
        self.data_labels = self.items[:, 2: config.num_of_class+2] if df.shape[1] > 2 else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        input_ids = self.data_encodings['input_ids'][index]
        token_type_ids = self.data_encodings['token_type_ids'][index]
        attention_masks = self.data_encodings['attention_mask'][index]

        label_ids = []
        if self.data_labels is not None:
            labels = self.data_labels[index]
            for label in labels:
                label_ids.append(float(label))

        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(token_type_ids, dtype=torch.long), \
               torch.tensor(attention_masks, dtype=torch.long),\
               torch.tensor(label_ids, dtype=torch.float)








