# Active Learning Baseline 使用说明

## 概述

`active_learning_baseline` 是一个专门为 `CNN_baseline` 模型设计的active learning函数，使用基于Bayesian inference的predictive covariance acquisition functions，而不是MC dropout方法。

## 两种Acquisition Functions

### 1. `predictive_covariance_mfvi` (Mean Field Variational Inference)
- 基于Lemma 1
- 通过最大化ELBO来找到最优的mean-field参数
- 需要迭代优化（较慢但更灵活）

### 2. `predictive_covariance_analytic` (Analytic Inference)
- 基于Lemma 2
- 直接计算matrix normal posterior的解析解
- 计算更快（推荐使用）

## 使用方法

### 基本使用

```python
from active_learning.active_train_loop_baseline import active_learning_baseline
from torchvision import datasets, transforms

# 加载数据
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 使用 Analytic 方法（推荐，更快）
model, history = active_learning_baseline(
    dataset=full_dataset,
    acquisition_fn="predictive_covariance_analytic",
    split_size=(20, 100, 10000),
    random_split=False,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100,
    K=10,
)

# 使用 MFVI 方法
model, history = active_learning_baseline(
    dataset=full_dataset,
    acquisition_fn="predictive_covariance_mfvi",
    split_size=(20, 100, 10000),
    random_split=False,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100,
    K=10,
    num_iterations=100,  # MFVI优化迭代次数
    lr=0.01,            # MFVI优化学习率
)
```

### 参数说明

**必需参数:**
- `dataset`: 数据集对象
- `acquisition_fn`: `"predictive_covariance_mfvi"` 或 `"predictive_covariance_analytic"`

**可选参数:**
- `split_size`: 数据分割大小 `(n_initial_train, n_val, n_test)` (默认 `(20, 100, 10000)`)
- `random_split`: 是否使用随机分割 (默认 `False`，使用固定seed=0)
- `weight_decay_candidates`: weight decay候选值列表 (默认 `[0.000001, 0.00001, 0.0001]`)
- `n_rounds`: acquisition轮数 (默认 `100`)
- `K`: 每轮acquire的数据点数 (默认 `10`)
- `sigma_sq`: 噪声方差 σ² (默认 `1.0`)
- `s_sq`: Prior方差 s² (默认 `1.0`)
- `num_iterations`: MFVI的ELBO优化迭代次数 (默认 `100`)
- `lr`: MFVI的ELBO优化学习率 (默认 `0.01`)
- `device`: 计算设备 (默认 `"cuda"`)

## 与原始 `active_learning` 的区别

| 特性 | `active_learning` | `active_learning_baseline` |
|------|-------------------|---------------------------|
| **模型** | `CNN` (有dropout) | `CNN_baseline` (无dropout) |
| **Acquisition** | MC dropout方法 | Bayesian inference方法 |
| **Acquisition函数** | entropy, BALD, variation_ratio, mean_std, random | predictive_covariance_mfvi, predictive_covariance_analytic |
| **MC参数** | 需要 `MC_acquire`, `MC_test`, `T` | 不需要MC参数 |

## 完整示例

```python
import torch
from torchvision import datasets, transforms
from active_learning.active_train_loop_baseline import active_learning_baseline

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 运行active learning
model, history = active_learning_baseline(
    dataset=full_dataset,
    acquisition_fn="predictive_covariance_analytic",  # 或 "predictive_covariance_mfvi"
    split_size=(20, 100, 10000),
    random_split=False,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100,
    K=10,
    sigma_sq=1.0,
    s_sq=1.0,
    device=device,
)

print(f"Final test accuracy: {history[-1]}")
print(f"History: {history}")
```

## 注意事项

1. **模型要求**: 必须使用 `CNN_baseline` 模型（函数内部自动使用）
2. **不使用MC dropout**: 所有测试和acquisition都不使用MC dropout
3. **计算复杂度**: 
   - `predictive_covariance_analytic`: 较快，推荐使用
   - `predictive_covariance_mfvi`: 需要优化ELBO，较慢
4. **数值稳定性**: 代码中已添加正则化项确保矩阵求逆的稳定性
5. **特征维度**: 假设特征维度K=128（CNN_baseline的fc1输出维度）

## 理论依据

两种方法都基于Bayesian linear regression来估算最后一层权重的posterior，然后计算predictive covariance作为acquisition score。根据提供的Lemma 1和Lemma 2，两种方法在理论上等价（使用trace或determinant作为acquisition function）。

