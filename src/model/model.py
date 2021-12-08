import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForTokenClassification
from torch import nn
from src.model.custom_loss import AsymmetricLoss

import torch

# TODO: Use transformers trained with Twitter Data
# TODO: Use official val_set
# TODO: Use word representations
class MEMbErt(nn.Module):
    def __init__(self):
        super(MEMbErt, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', add_prefix_space=False)
        self.tc = RobertaForTokenClassification.from_pretrained('roberta-large')
        self.fc = nn.Sequential(nn.Linear(in_features=2, out_features=20),
                                nn.Sigmoid())

    def forward(self, x):
        bpe_tokens = self.tokenizer(x, return_tensors='pt')
        tc_pretrained = self.tc(bpe_tokens['input_ids'][:, 1:-1], bpe_tokens['attention_mask'][:, 1:-1],
                                return_dict=False)
        out = self.fc(tc_pretrained[0])

        return out


def train(input_text: list, labels: list):

    train_val_ratio = 0.8
    train_memes, val_memes = train_test_split(list(zip(input_text, labels)),
                                              train_size=train_val_ratio,
                                              random_state=0)
    input_train, output_train = list(zip(*train_memes))
    input_val, output_val = list(zip(*val_memes))

    model = MEMbErt()
    loss_criterion = nn.BCELoss()  # Binary Cross-Entropy
    #loss_criterion = AsymmetricLoss()
    optimizer = Adam(model.parameters())
    num_epochs = 5
    batch_size = 1
    batch_per_epoch = len(input_train)
    model.train()

    for epoch in range(1, num_epochs + 1):
        with tqdm(range(batch_per_epoch), ncols=150, unit=' batch') as tepoch:
            t_loss_list = []
            for batch_idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch: {epoch}\t")
                output_t = torch.Tensor(output_train[batch])
                optimizer.zero_grad()
                prediction = model(' '.join(input_train[batch]))
                output_t = adjust_output(model.tokenizer, input_train[batch], output_t)
                loss = loss_criterion(prediction, output_t)
                t_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'loss': np.mean(t_loss_list)})

                if batch_idx + 1 == batch_per_epoch:
                    with torch.no_grad():
                        val_loss_list = []
                        for val_idx in range(len(input_val)):
                            output_tv = torch.Tensor(output_val[val_idx])
                            prediction = model(' '.join(input_val[val_idx]), output_tv)
                            loss = loss_criterion(prediction, output_tv)
                            val_loss_list.append(loss.item())

                        tepoch.set_postfix({'loss': np.mean(t_loss_list),
                                            'val_loss': np.mean(val_loss_list)})

    torch.save(model, 'models/MEMbErt.pth')
