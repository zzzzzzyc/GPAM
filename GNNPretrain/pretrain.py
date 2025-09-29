# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : main(pretrain).py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from Main.pargs import pargs
from Main.dataset_uweibo import TreeDataset
from Main.model import ResGCN_graphcl, BiGCN_graphcl, GraphSAGE
from Main.augmentation import augment
from loss.contrastive_loss import ContrastiveLoss, GraceLoss


def pre_train(dataloader, aug1, aug2, model, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    pbar = tqdm(total=len(dataloader))
    for step,data in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        data.x = data.x.type(torch.bfloat16)
        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)
        z1 = model.forward_graphcl(aug_data1)
        z2 = model.forward_graphcl(aug_data2)

        proj_z1 = model.projection(z1)
        proj_z2 = model.projection(z2)

        principal_component = all_principal_component
        loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
        total_ins_loss += ins_loss * proj_z1.shape[0]
        total_con_loss += contrast_loss * proj_z1.shape[0]
        total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]

        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print('Step {:05d} | Loss {:.4f} | Instance Loss {:.4f} | Contrastive Loss {:.4f}'.format(step, loss.item(), ins_loss, contrast_loss))

        pbar.update()

        # total_loss += loss.item() * data.num_graphs

    total_mean_loss = total_loss / len(dataloader.dataset)
    total_mean_instance_loss = total_ins_loss / len(dataloader.dataset)
    total_mean_contrastive_loss = total_con_loss / len(dataloader.dataset)

    return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss



if __name__ == '__main__':
    args = pargs()

    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    batch_size = args.batch_size
    undirected = args.undirected
    centrality = args.centrality
    weight_decay = args.weight_decay
    epochs = args.epochs

    unlabel_dataset_path = "path/UWeibo/dataset"

    unlabel_dataset = TreeDataset(unlabel_dataset_path, centrality, undirected)
    unsup_train_loader = DataLoader(unlabel_dataset, batch_size, shuffle=True)

    num_classes = 4 if 'Twitter' in dataset or dataset == 'PHEME' else 2
    if args.model == 'ResGCN':
        model = ResGCN_graphcl(dataset=unlabel_dataset, num_classes=num_classes, hidden=args.hidden,out_feats = args.num_out,
                               num_proj_hidden=args.num_out,
                               num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv,
                               num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                               residual=args.skip_connection, res_branch=args.res_branch,
                               global_pool=args.global_pool, dropout=args.dropout,
                               edge_norm=args.edge_norm).to(device)
        model = model.to(dtype=torch.bfloat16)
    elif args.model == 'BiGCN':
        model = BiGCN_graphcl(unlabel_dataset.num_features, hid_feats=args.hidden, out_feats=args.num_out,
                              num_proj_hidden=args.num_out, num_classes=num_classes).to(device)
        model = model.to(dtype=torch.bfloat16)

    elif args.model == 'SAGE':
        model = GraphSAGE(
            unlabel_dataset.num_features,
            hidden_channels=args.hidden,
            out_channels=args.num_out,
            n_layers=args.num_layers,
            num_proj_hidden=args.num_out,
            activation=F.relu,
            dropout=args.dropout,
            edge_dim=None,
            gnn_type=args.gnn_type
        ).to(device)
        model = model.to(dtype=torch.bfloat16)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    all_principal_component = torch.load('./PCA_1000_pc_internVL2_5_8B.pt').to(device, dtype=torch.bfloat16)

    criterion = ContrastiveLoss(args.tau, self_tp=args.self_tp).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    no_increase = 0
    best_epoch=0
    best_loss = 1000000000
    if args.load_parameters == False:
        checkpoint_path = './saved_model/SAGE_pretrain.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model weights loaded.")


        for epoch in range(1, epochs + 1):
            total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss = pre_train(unsup_train_loader,
                                                                                               args.aug1, args.aug2,
                                                                                               model, optimizer,
                                                                                               criterion,epoch,
                                                                                               device)
            if total_mean_loss < best_loss:
                best_loss = total_mean_loss
                best_epoch = epoch
                no_increase = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                no_increase += 1
                if no_increase > args.patience:
                    break
        # model.load_state_dict(torch.load('./saved_model/BiGCN_graphcl_pretrain.pth'))

    else:
        model.load_state_dict(torch.load('./saved_model/ResGCN_pretrain.pth'))













