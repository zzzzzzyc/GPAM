import warnings
import sys
import os
import os.path as osp

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))


import json
import os
import re
from collections import defaultdict
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import datetime
import numpy as np
from tqdm import tqdm
from Main.dataset_weibo import TreeDataset

def normalize(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def time_diff_score(source_time, comment_time,Twitter=False):
    if Twitter:
        fmt = "%a %b %d %H:%M:%S %z %Y"
        t_source = datetime.datetime.strptime(source_time, fmt)
        t_comment = datetime.datetime.strptime(comment_time, fmt)
        diff_sec = abs((t_comment - t_source).total_seconds())
    else:
        fmt_standard = '%Y-%m-%d %H:%M:%S'
        fmt_chinese = '%m月%d日 %H:%M'
        try:
            t_source = datetime.datetime.strptime(source_time, fmt_standard)
        except ValueError:
            t_source = datetime.datetime.strptime('2013年' + source_time, '%Y年' + fmt_chinese)
        try:
            t_comment = datetime.datetime.strptime(comment_time, fmt_standard)
        except ValueError:
            t_comment = datetime.datetime.strptime('2013年' + comment_time, '%Y年' + fmt_chinese)

        diff_sec = abs((t_comment - t_source).total_seconds())

    return np.exp(-diff_sec / 3600)

def save_text(new_data, keep_indices, post_json, save_dir='structured_texts'):
    os.makedirs(save_dir, exist_ok=True)
    tweet_id = post_json['source']['tweet id']

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

    texts = []
    for new_idx, old_idx in enumerate(keep_indices):
        if old_idx == 0:
            continue
        cid = old_idx - 1
        text = post_json['comment'][cid]['content'] if cid < len(post_json['comment']) else '[UNK]'
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        texts.append({"id": new_idx, "text": text})

    id_to_text = {item["id"]: item["text"] for item in texts}

    children_map = defaultdict(list)
    for src, dst in new_data.edge_index.t().tolist():
        if src in old_to_new and dst in old_to_new:
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            if new_src != 0:
                children_map[new_src].append(new_dst)

    all_nodes = set(id_to_text.keys())
    child_nodes = set(c for v in children_map.values() for c in v)
    root_nodes = sorted(all_nodes - child_nodes)

    lines = []

    def dfs(node_id, prefix, depth):
        indent = "  " * depth
        lines.append(f"{indent}{prefix} {id_to_text.get(node_id, '[UNK]')}")
        for i, child in enumerate(children_map.get(node_id, []), 1):
            dfs(child, f"{prefix}.{i}", depth + 1)

    for i, root in enumerate(root_nodes, 1):
        dfs(root, str(i), 0)

    save_path = os.path.join(save_dir, f"{tweet_id}.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def retrive_on_graphs(post_json, data, sim_weight=0.8, time_weight=0.2, centrality_weight=0.2, top_k=10):
    x = data.x
    x = F.normalize(x, p=2, dim=1)
    root_embed = x[0].unsqueeze(0)

    sim_scores = [1.0]
    time_scores = [1.0]
    # centrality_raw = post_json['centrality']['Pagerank']
    # centrality_scores = [centrality_raw[0] if len(centrality_raw) > 0 else 0]

    for i, comment in enumerate(post_json['comment']):
        idx = comment['comment id'] + 1
        sim = F.cosine_similarity(root_embed, x[idx].unsqueeze(0)).item()
        sim_scores.append(sim)

        t_score = time_diff_score(post_json['source']['time'], comment['time'], Twitter=True)
        time_scores.append(t_score)

        # c_score = centrality_raw[idx] if idx < len(centrality_raw) else 0
        # centrality_scores.append(c_score)

    sim_scores = normalize(sim_scores)
    time_scores = normalize(time_scores)
    # centrality_scores = normalize(centrality_scores)

    total_scores = sim_weight * np.array(sim_scores) + \
                   time_weight * np.array(time_scores)
    topk_indices = np.argsort(-total_scores)[:top_k+1]
    keep_indices = sorted(topk_indices.tolist())

    return keep_indices


if __name__ == '__main__':
    root_path = 'path/shell/data/Weibo'
    dataset_path = 'path/Weibo/dataset'
    processed_dir = os.path.join(dataset_path, 'processed')
    raw_dir = os.path.join(dataset_path, 'raw')
    # data_file = os.path.join(processed_dir, 'data.pt')

    finetune = True
    centrality = 'Pagerank'
    # data, slices = torch.load(data_file, weights_only=False)
    dataset = TreeDataset(dataset_path,centrality, undirected=False)
    # dataset.data, dataset.slices = data, slices


    save_dir = os.path.join(root_path, 'graphs')
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        ori_data = Data(
            x=data.x.clone().detach(),
            edge_index=data.edge_index.clone().detach(),
            y=data.y.clone().detach() if data.y is not None else None,
            no_root_edge_index=data.no_root_edge_index.clone().detach() if hasattr(data,'no_root_edge_index') else None,
            centrality=data.centrality.clone().detach() if hasattr(data, 'centrality') else None
        )
        raw_file = dataset.raw_file_names[i]
        post_json = json.load(open(os.path.join(dataset.raw_dir, raw_file), 'r', encoding='utf-8'))
        keep_indices= retrive_on_graphs(post_json, ori_data, top_k=5)
        tweet_id = os.path.splitext(raw_file)[0]
        save_path = os.path.join(save_dir, f'{tweet_id}.pt')
        torch.save(ori_data, save_path)
        if finetune:
            save_text(ori_data, keep_indices, post_json, save_dir='path/shell/data/weibo/text_graph')
