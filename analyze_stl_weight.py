import pandas as pd
import glob
import os

# 定义 stl_weight 值
stl_weights = ['0.01', '0.05', '0.1', '0.5', '1']

results = {}

for weight in stl_weights:
    # 找到对应的结果目录
    pattern = f"result/label_hparam_*_stl_weight_s{weight.replace('.', 'p')}/*/*.csv"
    files = glob.glob(pattern)
    
    if not files:
        continue
    
    all_aff_f = []
    dataset_scores = {}
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'affiliation_f' in df.columns:
                aff_f = df['affiliation_f'].mean()
                all_aff_f.append(aff_f)
                
                # 提取数据集名称
                dataset = os.path.basename(os.path.dirname(file))
                if dataset not in dataset_scores:
                    dataset_scores[dataset] = []
                dataset_scores[dataset].append(aff_f)
        except Exception as e:
            continue
    
    if all_aff_f:
        avg_aff_f = sum(all_aff_f) / len(all_aff_f)
        results[weight] = {
            'avg_aff_f': avg_aff_f,
            'dataset_scores': {k: sum(v)/len(v) for k, v in dataset_scores.items()}
        }

# 打印结果
print("=" * 60)
print("STL Weight Sensitivity Analysis (Aff-F)")
print("=" * 60)

for weight in stl_weights:
    if weight in results:
        print(f"\nSTL Weight = {weight}:")
        print(f"  Average Aff-F: {results[weight]['avg_aff_f']:.4f}")
        print(f"  Per-dataset:")
        for dataset, score in sorted(results[weight]['dataset_scores'].items()):
            print(f"    {dataset}: {score:.4f}")

# 找出最好的
if results:
    best_weight = max(results.keys(), key=lambda x: results[x]['avg_aff_f'])
    print(f"\n" + "=" * 60)
    print(f"BEST: STL Weight = {best_weight}, Aff-F = {results[best_weight]['avg_aff_f']:.4f}")
    print("=" * 60)
