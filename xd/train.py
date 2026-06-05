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
        s1, s2, s3 = out["video_scores"]

        loss_main = criterion(s3, labels)
        loss1 = criterion(s1, labels)
        loss2 = criterion(s2, labels)

        h_input, h_out, h_c, h_s, gate = out["csfd"]
        # L_{CSFD} - 可开关
        if args.enable_csfd_loss and h_c is not None and h_s is not None:
            L_orth, L_sparse = compute_csfd_loss(h_c, h_s, h_out, h_input)
            loss_aux2 = 0.2 * L_orth + 0.05 * L_sparse
        else:
            loss_aux2 = 0.0

        loss_aux1 = loss1 + loss2

        loss = loss_main + args.gamma * loss_aux1 + loss_aux2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)

    return sum(total_loss) / len(total_loss)
