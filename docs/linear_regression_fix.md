# 线性回归模型偏置更新修复

## 问题描述

在 `src/chap02_linear_regression/linear_regression-tf2.0.py` 文件中，存在两个bug：

### Bug 1：偏置项未被更新

在 `train_one_step` 函数中，只计算了权重 `model.w` 的梯度并更新，但完全没有计算和更新偏置 `model.b`。

**问题代码：**
```python
grads = tape.gradient(loss, model.w)    # 只对 w 求梯度
optimizer.apply_gradients([(grads, model.w)])    # 只更新 w
```

**修复后：**
```python
grads = tape.gradient(loss, [model.w, model.b])    # 对 w 和 b 求梯度
optimizer.apply_gradients(zip(grads, [model.w, model.b]))    # 更新 w 和 b
```

### Bug 2：图例标签不匹配

第200行的图例有3个标签 `["train", "test", "pred"]`，但实际只绘制了2条曲线（训练数据点和测试预测线）。

**修复后：**
```python
plt.legend(["train", "pred"])  # 与实际绘制的曲线对应
```

## 影响分析

### 偏置未更新的影响

1. **模型表达能力受限**：缺少偏置项，模型只能学习通过原点的线性关系
2. **预测精度下降**：对于数据分布不经过原点的情况，模型无法准确拟合
3. **训练不稳定**：偏置项有助于模型收敛，缺少它可能导致训练过程不稳定

### 图例不匹配的影响

1. **可视化混淆**：图例与实际曲线不对应，影响结果理解
2. **专业性降低**：在学术或工程报告中使用时显得不专业

## 修复验证

修复后的代码可以参考改进版文件 `linear_regression-tf2.0-improved.py`，其中已经包含了正确的偏置更新逻辑。

## 相关文件

- 原始文件：`src/chap02_linear_regression/linear_regression-tf2.0.py`
- 改进版文件：`src/chap02_linear_regression/linear_regression-tf2.0-improved.py`
