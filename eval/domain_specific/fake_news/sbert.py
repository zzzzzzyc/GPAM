import os
import json
import torch
import argparse
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = '/home/hpclp/disk/q/models/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_type, device=f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    sbert_embeds = model.encode(texts, batch_size=16, show_progress_bar=True)
    return torch.tensor(sbert_embeds)


def evaluate_sbert_similarity(data, device=0):
    gt_texts = [item['answer'] for item in data]
    pred_texts = [item['response'] for item in data]

    # Step 1: Get embeddings
    gt_embed = get_sbert_embedding("sbert", gt_texts, device)
    pred_embed = get_sbert_embedding("sbert", pred_texts, device)

    # Step 2: Normalize and calculate prediction similarity
    gt_embed = F.normalize(gt_embed, p=2, dim=1)
    pred_embed = F.normalize(pred_embed, p=2, dim=1)
    sbert_score = (gt_embed * pred_embed).sum(1).mean().item()

    # Step 3: Base value from all reasoning pair combinations (GT + Pred)
    all_texts = gt_texts + pred_texts
    all_embed = get_sbert_embedding("sbert", all_texts, device)
    all_embed = F.normalize(all_embed, p=2, dim=1)
    sim_matrix = torch.mm(all_embed, all_embed.T)
    sim_matrix.fill_diagonal_(0)
    n = sim_matrix.size(0)
    base_value = (sim_matrix.sum() / (n * (n - 1))).item()

    return sbert_score, base_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the result JSON file')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    sbert_score, base_value = evaluate_sbert_similarity(data, device=args.device)

    result = {
        'sbert_score': sbert_score,
        'base_value': base_value,
        'outputs': data
    }

    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=4)
