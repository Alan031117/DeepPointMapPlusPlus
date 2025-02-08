# DeepPointMapPlusPlus

Source code for Accurate and Efficient LiDAR SLAM by Learning Unified Neural Descriptors.

## Training

```bash
python pipeline/train.py --yaml_file my_yaml.yaml --gpu_index 0
```

## Inference

```bash
python pipeline/infer.py --yaml_file my_yaml.yaml --gpu_index 0 --weight my_weight.pth