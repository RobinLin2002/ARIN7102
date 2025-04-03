import json
from datasets import load_dataset  
# 加载数据集
ds = load_dataset('FreedomIntelligence/huatuo_encyclopedia_qa', 
                 cache_dir='/home/Data1/ziyue/dataset')

# 保存为 JSON 文件
for split in ["train", "test", "validation"]:
    ds[split].to_json(f"{split}_data.json", orient="records", force_ascii=False)