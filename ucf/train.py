import torch
from tqdm import tqdm
from losses.loss import compute_csfd_loss


def train(train_loader, model, optimizer, criterion, args):
    model.train()

    total_loss = []
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
        inputs = inputs[:, :torch.max(seq_len), :]
        inputs = inputs.float().to(args.device)
        labels = labels.float().to(args.device)

        out = model(inputs, seq_len)
        s1, s2 = out["video_scores"]

        loss_main = criterion(s2, labels)
        loss1 = criterion(s1, labels)

        # h_input, h_out, h_c, h_s, gate = out["csfd"]
        # L_orth, L_sparse = compute_csfd_loss(h_c, h_s, h_out, h_input)

        loss_aux1 = loss1
        # loss_aux2=0.2*L_orth + 0.05*L_sparse

        # loss = loss_main + args.gamma * loss_aux1 + loss_aux2
        # loss = loss_main + args.gamma * loss_aux1
        loss = loss_main + loss_aux1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)

    return sum(total_loss) / len(total_loss)
