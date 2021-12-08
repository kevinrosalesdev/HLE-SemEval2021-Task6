import numpy as np
import torch

from torch.optim.adam import Adam
from torch import nn
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForTokenClassification
from src.model.custom_loss import AsymmetricLoss


# TODO: Use transformers trained with Twitter Data
# TODO: Use word representations
class MEMbErt(nn.Module):
    def __init__(self):
        super(MEMbErt, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=False)
        self.tc = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=20)
        # self.fc = nn.Linear(in_features=2, out_features=20)
        self.act = nn.Sigmoid()

    def forward(self, x):
        bpe_tokens = self.tokenizer(x, return_tensors='pt')
        tc_pretrained = self.tc(input_ids=bpe_tokens['input_ids'][:, 1:-1].cuda(),
                                attention_mask=bpe_tokens['attention_mask'][:, 1:-1].cuda(),
                                return_dict=False)
        out = self.act(tc_pretrained[0])
        # out = self.fc(tc_pretrained[0])

        return out.squeeze(0)


def train(input_t_text: list, t_labels: list, input_v_text: list, v_labels: list):

    model = MEMbErt().cuda()
    loss_criterion = nn.BCELoss()  # Binary Cross-Entropy
    # loss_criterion = AsymmetricLoss()
    optimizer = Adam(model.parameters())
    num_epochs = 5
    batch_size = 1
    batch_per_epoch = len(input_t_text)
    model.train()

    for epoch in range(1, num_epochs + 1):
        with tqdm(range(batch_per_epoch), ncols=150, unit=' batch') as tepoch:
            t_loss_list = []
            for batch_idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch: {epoch}\t")
                output_t = torch.Tensor(t_labels[batch])
                optimizer.zero_grad()
                sentence = ' '.join(input_t_text[batch])
                prediction = model(sentence)
                # prediction = adjust_output(model.tokenizer, sentence, prediction)
                # prediction.requires_grad = True
                output_t = adjust_gt(model.tokenizer, sentence, output_t).cuda()
                loss = loss_criterion(prediction, output_t)
                t_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'loss': np.mean(t_loss_list)})

                if batch_idx + 1 == batch_per_epoch:
                    with torch.no_grad():
                        val_loss_list = []
                        for val_idx in range(len(input_v_text)):
                            output_tv = torch.Tensor(v_labels[val_idx])
                            sentence = ' '.join(input_v_text[val_idx])
                            prediction = model(sentence)
                            # prediction = adjust_output(model.tokenizer, sentence, prediction)
                            # prediction.requires_grad = True
                            output_tv = adjust_gt(model.tokenizer, sentence, output_tv).cuda()
                            loss = loss_criterion(prediction, output_tv)
                            val_loss_list.append(loss.item())

                        tepoch.set_postfix({'loss': np.mean(t_loss_list),
                                            'val_loss': np.mean(val_loss_list)})

    torch.save(model, 'models/MEMbErt.pth')


def adjust_gt(tokenizer: RobertaTokenizer, sentence: str, output: torch.Tensor) -> torch.Tensor():
    tokenizer_match = tokenizer.tokenize(sentence)
    output = output.detach().numpy()
    res = [output[0]]
    counter = 1
    for token in tokenizer_match[1:]:
        if token[0] == 'Ġ':
            res.append(output[counter])
            counter += 1
        else:
            res.append(res[-1])

    return torch.from_numpy(np.array(res))



