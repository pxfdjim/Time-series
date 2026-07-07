import pandas as pd
import glob
import os

# 分析 stl_weight=0 的结果
pattern = "result/label_hparam_*_stl_weight_s0/*/*.csv"
files = glob.glob(pattern)

all_aff_f = []
dataset_scores = {}

for file in files:
    try:
        df = pd.read_csv(file)
        if 'affiliation_f' in df.columns:
            aff_f = df['affiliation_f'].mean()
            all_aff_f.append(aff_f)
            
            dataset = os.path.basename(os.path.dirname(file))
            if dataset not in dataset_scores:
                dataset_scores[dataset] = []
            dataset_scores[dataset].append(aff_f)
    except Exception as e:
        continue

print("=" * 60)
print("STL Weight = 0 (No STL Loss)")
print("=" * 60)

if all_aff_f:
    avg_aff_f = sum(all_aff_f) / len(all_aff_f)
    print(f"\nAverage Aff-F: {avg_aff_f:.4f}")
    print(f"Per-dataset:")
    for dataset, scores in sorted(dataset_scores.items()):
        avg_score = sum(scores) / len(scores)
        print(f"  {dataset}: {avg_score:.4f}")
    
    # 计算百分比
    print(f"\nConverted to percentages:")
    for dataset, scores in sorted(dataset_scores.items()):
        avg_score = sum(scores) / len(scores)
        print(f"  {dataset}: {avg_score * 100:.2f}")
    print(f"\nAverage: {avg_aff_f * 100:.2f}")
