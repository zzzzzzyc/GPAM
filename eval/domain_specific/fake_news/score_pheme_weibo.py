import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  precision_recall_fscore_support

def normalize_response(resp: str):
    resp = resp.lower()
    if 'yes' in resp:
        return 'yes'
    elif 'no' in resp:
        return 'no'
    else:
        return 'no'

def evaluate(results):
    y_true = []
    y_pred = []

    for item in results:
        gt_raw = item['answer']
        pred_raw = item['response']

        # gt_first_word = gt_raw.strip().split('.')[0].lower()
        # gt = 'yes' if gt_first_word == 'yes' else 'no'

        # first_word = pred_raw.strip().strip().split('.')[0].strip().lower()
        # pred = 'fake' if first_word == 'fake' else 'true'

        # last_line_pred = pred_raw.strip().split('\n')[-1].strip()  # 最后一行
        # last_word_pred = last_line_pred.split()[-1].rstrip('.').lower()  # 最后一个单词
        # pred = 'fake' if last_word_pred == 'fake' else 'true'
        # pred_lower = pred_raw.lower()
        # if 'fake' in pred_lower:
        #     pred = 'fake'
        # elif 'true' in pred_lower:
        #     pred = 'true'
        # else:
        #     pred = 'true'  # 或 raise 异常

        # gt = item['answer'].strip().capitalize()  # ground truth: "Yes"/"No"
        # first_word_gt = gt_raw.strip().strip().split('.')[0].strip().lower()
        # gt = 'fake' if first_word_gt == 'fake' else 'true'
        # last_line_gt = gt_raw.strip().split('\n')[-1].strip()
        # last_word_gt = last_line_gt.split()[-1].rstrip('.').lower()
        # gt = 'fake' if last_word_gt == 'fake' else 'true'

        # gt = normalize_response(gt_raw)
        # pred = normalize_response(pred_raw)

        y_true.append(1 if gt_raw == 'no' else 0)
        y_pred.append(1 if pred_raw == 'no' else 0)

        # y_true.append(1 if gt == 'fake' else 0)
        # y_pred.append(1 if pred == 'fake' else 0)


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'Number of samples evaluated: {len(y_true)}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    precision_all, recall_all, f1_all, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )

    print("\nPer-class metrics:")
    print("True News (No):")
    print(f"  Precision = {precision_all[0]:.4f}")
    print(f"  Recall    = {recall_all[0]:.4f}")
    print(f"  F1-score  = {f1_all[0]:.4f}")

    print("Fake News (Yes):")
    print(f"  Precision = {precision_all[1]:.4f}")
    print(f"  Recall    = {recall_all[1]:.4f}")
    print(f"  F1-score  = {f1_all[1]:.4f}")


    return acc, prec, rec ,f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']
    acc, prec, rec, f1 = evaluate(data)

    results = {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'outputs': data
    }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
