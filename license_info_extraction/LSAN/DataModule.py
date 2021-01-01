import nltk
import pandas as pd
from torch.utils.data import Dataset

from config import *
from utils import *


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, word_embeddings):
        df = pd.read_csv(dataset_file_path, header=0, encoding='unicode_escape')
        self.items = df.values
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        text_id = self.items[index][0]
        text = self.items[index][1]
        labels = self.items[index][2: num_of_class+2]

        # pre-processing input text
        text = text.lower().strip()
        text = filter_sentence(text)

        tokens = nltk.word_tokenize(text)
        # print(tokens)
        # print(len(tokens))

        token_to_ids = []
        for token in tokens:
            try:
                token_to_ids.append(self.word_embeddings.stoi[token])
            except KeyError:
                token_to_ids.append(self.word_embeddings.stoi['<unk>'])
        # print(token_to_ids)

        padding_length = MAX_SEQ_LENGTH - len(tokens)
        token_to_ids = token_to_ids + [self.word_embeddings.stoi['<pad>']] * padding_length

        assert len(token_to_ids) == MAX_SEQ_LENGTH

        label_ids = []
        for label in labels:
            label_ids.append(float(label))
        # print(label_ids)

        return torch.tensor(token_to_ids, dtype=torch.long), \
               torch.tensor(label_ids, dtype=torch.float)


