# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
import torch.nn.functional as F


class TreeDataset(InMemoryDataset):
    def __init__(self, root, centrality_metric, undirected, transform=None, pre_transform=None,
                 pre_filter=None):
        self.centrality_metric = centrality_metric
        self.undirected = undirected
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return os.listdir(self.raw_dir)
    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        label_map = {
            "non-rumor": 0,
            "false": 1
        }

        for filename in raw_file_names:
            centrality = None
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))

            # node_texts = []
            #
            # source_text = post['source']['content']
            tweet_id = post['source']['tweet id']
            news_text_path = '/home/hpclp/disk/Graphgpt/DPSG-main/DPSG/data/PHEME/text_embeddings/tweet_text/'
            f = open(news_text_path + tweet_id + '.txt', 'r')
            content = np.loadtxt(f, delimiter = ' ')
            x = torch.tensor(content, dtype=torch.float32).view(1, -1)
            x = F.normalize(x, p=2, dim=1)
            # node_texts.append((0, source_text))  # 节点 0 是源节点

            if 'label' in post['source'].keys():
                label_str = post['source']['label']
                if label_str in label_map:
                    y.append(label_map[label_str])
            for i, comment in enumerate(post['comment']):

                comment_id = comment['comment id'] + 1  # 从1开始编号
                # comment_text = comment['content']
                # node_texts.append((comment_id, comment_text))

                post_id = comment['origin id']
                post_text_path = '/home/hpclp/disk/Graphgpt/DPSG-main/DPSG/data/PHEME/text_embeddings/tweet_text/'
                f = open(post_text_path + post_id + '.txt', 'r')
                content = np.loadtxt(f, delimiter=' ')
                content = torch.tensor(content, dtype=torch.float32).view(1, -1)
                content = F.normalize(content, p=2, dim=1)
                x = torch.cat([x, content], 0)
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            if self.centrality_metric == "Degree":
                centrality = torch.tensor(post['centrality']['Degree'], dtype=torch.float32)
            elif self.centrality_metric == "PageRank":
                centrality = torch.tensor(post['centrality']['Pagerank'], dtype=torch.float32)
            elif self.centrality_metric == "Eigenvector":
                centrality = torch.tensor(post['centrality']['Eigenvector'], dtype=torch.float32)
            elif self.centrality_metric == "Betweenness":
                centrality = torch.tensor(post['centrality']['Betweenness'], dtype=torch.float32)
            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = to_undirected(torch.LongTensor(edge_index)) if self.undirected else torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)
            one_data = Data(x=x, y=y, edge_index=edge_index, no_root_edge_index=no_root_edge_index,
                            centrality=centrality) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index, centrality=centrality)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])
