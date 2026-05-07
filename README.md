# CNN 与 ViT 在不同数据规模下的 CIFAR-10 对比实验

本项目用于比较 CNN 和 Vision Transformer 在 CIFAR-10 64x64 数据集上的表现，重点观察 10%、20%、100% 训练数据规模下，两类模型的准确率、F1-score、训练曲线和可视化结果。

当前版本已经按队员 B 的任务要求完成代码整理：统一配置、统一入口、统一日志、统一 checkpoint 命名，并修复 attention 可视化脚本的 ViT 参数不一致问题。

## 一、组员先看这里

队员 A 跑实验时，主要使用：

```bash
python main.py --model cnn --data_ratio 0.1 --epochs 100 --device cuda
python main.py --model vit --data_ratio 1.0 --epochs 100 --device cuda
```

队员 C 画图时，主要读取：

```text
logs/*.csv
checkpoints/*_best.pth
```

正式实验前请先跑 debug：

```bash
python main.py --model cnn --data_ratio 0.1 --epochs 2 --debug
python main.py --model vit --data_ratio 0.1 --epochs 2 --debug
```

debug 只检查代码能不能跑通，不代表正式结果。

## 二、项目结构

```text
config.py                  统一配置文件
main.py                    统一实验入口
train.py                   训练循环、日志、checkpoint 保存
run.sh                     一键运行六组主实验
models/cnn.py              CNN 模型
models/vit.py              ViT 模型
utils/dataset.py           数据加载与数据增强
visualize_attention.py     ViT attention map 可视化
plot_learning_curve.py     学习曲线绘图脚本
plot_tradeoff.py           速度和准确率权衡图脚本
logs/                      CSV 训练日志输出目录
checkpoints/               模型权重输出目录
figures/                   图表输出目录
```

## 三、环境安装

进入项目根目录后运行：

```bash
pip install -r requirements.txt
```

如果训练机器有 NVIDIA 显卡，建议安装 CUDA 版 PyTorch。先检查显卡驱动：

```bash
nvidia-smi
```

如果能看到 GPU 信息，再安装 CUDA 版 PyTorch：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

安装后验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果输出里 `torch.cuda.is_available()` 是 `True`，正式训练时就可以使用 `--device cuda`。

主要依赖包括：

```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
tqdm
```

## 四、配置文件怎么用

核心参数集中在 `config.py`。

常用默认值：

```text
num_epochs = 100
batch_size = 128
learning_rate = 3e-4
weight_decay = 0.05
scheduler = warmup_cosine
warmup_epochs = 5
min_lr = 1e-6
use_amp = True
label_smoothing = 0.1
grad_clip_norm = 1.0
```

ViT 默认参数：

```text
img_size = 64
patch_size = 8
embed_dim = 256
depth = 6
num_heads = 8
mlp_ratio = 4.0
dropout = 0.1
```

一般情况下，组员不需要手动修改 `train.py`。如果要临时调参数，优先通过命令行传参；如果是长期默认设置，再改 `config.py`。

## 五、如何运行单组实验

有 CUDA 的机器建议显式加 `--device cuda`，例如：

```bash
python main.py --model cnn --data_ratio 1.0 --epochs 100 --device cuda
python main.py --model vit --data_ratio 1.0 --epochs 100 --device cuda
```

没有 CUDA 的机器去掉 `--device cuda`，程序会在 CPU 上运行。

运行 CNN，使用 10% 训练数据：

```bash
python main.py --model cnn --data_ratio 0.1 --epochs 100
```

运行 CNN，使用 20% 训练数据：

```bash
python main.py --model cnn --data_ratio 0.2 --epochs 100
```

运行 CNN，使用 100% 训练数据：

```bash
python main.py --model cnn --data_ratio 1.0 --epochs 100
```

运行 ViT，使用 10% 训练数据：

```bash
python main.py --model vit --data_ratio 0.1 --epochs 100
```

运行 ViT，使用 20% 训练数据：

```bash
python main.py --model vit --data_ratio 0.2 --epochs 100
```

运行 ViT，使用 100% 训练数据：

```bash
python main.py --model vit --data_ratio 1.0 --epochs 100
```

## 六、如何一键运行全部主实验

六组主实验包括：

```text
CNN + 10% data
CNN + 20% data
CNN + 100% data
ViT + 10% data
ViT + 20% data
ViT + 100% data
```

运行：

```bash
bash run.sh
```

如果 Windows PowerShell 不能直接运行 `bash run.sh`，就手动复制第五节的六条命令依次运行。

## 七、debug 快速测试

正式训练前，建议先跑：

```bash
python main.py --model cnn --data_ratio 0.1 --epochs 2 --debug
python main.py --model vit --data_ratio 0.1 --epochs 2 --debug
```

debug 模式会使用少量样本，速度更快，用来检查：

```text
dataloader 是否正常
model 是否能前向传播
train loop 是否能跑通
CSV 是否能写入
checkpoint 是否能保存
```

debug 输出文件会带 `debug_` 前缀，不要把它当正式实验结果。

## 八、强数据增强怎么开

默认主实验不强制开启 strong augmentation，避免和主实验因素混在一起。

如果要额外测试强增强：

```bash
python main.py --model vit --data_ratio 1.0 --epochs 100 --strong_aug
```

当前 strong augmentation 包括：

```text
RandomCrop(64, padding=8)
RandomHorizontalFlip
ColorJitter(0.4, 0.4, 0.4, 0.1)
RandomErasing(p=0.25)
```

注意：本项目不做 Mixup / CutMix。

## 九、训练输出在哪里

每组正式实验会生成一个 CSV 日志：

```text
logs/cnn_ratio0.1.csv
logs/cnn_ratio0.2.csv
logs/cnn_ratio1.0.csv
logs/vit_ratio0.1.csv
logs/vit_ratio0.2.csv
logs/vit_ratio1.0.csv
```

CSV 字段包括：

```text
epoch
train_loss
train_acc
val_loss
val_acc
val_f1
lr
epoch_time
best_val_acc
model_type
data_ratio
use_amp
use_strong_aug
seed
```

每组正式实验会保存 best checkpoint：

```text
checkpoints/cnn_ratio0.1_best.pth
checkpoints/cnn_ratio0.2_best.pth
checkpoints/cnn_ratio1.0_best.pth
checkpoints/vit_ratio0.1_best.pth
checkpoints/vit_ratio0.2_best.pth
checkpoints/vit_ratio1.0_best.pth
```

同时还会保存最后一轮：

```text
checkpoints/cnn_ratio0.1_last.pth
checkpoints/vit_ratio1.0_last.pth
```

## 十、队员 A 如何记录结果

队员 A 跑完每组实验后，需要记录：

```text
model
data_ratio
epochs
learning_rate
batch_size
weight_decay
use_amp
use_strong_aug
best_val_acc
best_val_f1
best_epoch
log_path
checkpoint_path
notes
```

建议整理成 `results_summary.csv`。

如果实验报错，请反馈给队员 B：

```text
运行命令
完整报错信息
当前模型是 cnn 还是 vit
当前 data_ratio
是否使用 --strong_aug
是否使用 --debug
```

不要只说“跑不了”，要给出完整命令和报错。

## 十一、队员 C 如何画图

学习曲线输入：

```text
logs/cnn_ratio0.1.csv
logs/cnn_ratio0.2.csv
logs/cnn_ratio1.0.csv
logs/vit_ratio0.1.csv
logs/vit_ratio0.2.csv
logs/vit_ratio1.0.csv
```

数据规模趋势图输入：

```text
每个 CSV 里的 best_val_acc 或 val_f1
```

混淆矩阵输入：

```text
checkpoints/cnn_ratio1.0_best.pth
checkpoints/vit_ratio1.0_best.pth
```

建议输出：

```text
figures/learning_curve_cnn.png
figures/learning_curve_vit.png
figures/data_scale_trend_accuracy.png
figures/data_scale_trend_f1.png
figures/confusion_matrix_cnn_ratio1.0.png
figures/confusion_matrix_vit_ratio1.0.png
```

## 十二、如何生成 ViT attention map

使用正式训练好的 ViT checkpoint：

```bash
python visualize_attention.py --checkpoint checkpoints/vit_ratio1.0_best.pth --max_images 3
```

输出示例：

```text
figures/attention_map_vit_ratio1.0_sample1.png
figures/attention_map_vit_ratio1.0_sample2.png
figures/attention_map_vit_ratio1.0_sample3.png
```

如果要指定某张图片：

```bash
python visualize_attention.py --checkpoint checkpoints/vit_ratio1.0_best.pth --image path/to/image.png
```

attention 脚本会从 checkpoint 或 `config.py` 读取 ViT 参数，避免 `patch_size`、`embed_dim`、`depth`、`num_heads` 不一致导致加载失败。

## 十三、训练优化说明

当前训练流程已经加入：

```text
AdamW optimizer
learning_rate = 3e-4
weight_decay = 0.05
Warmup + Cosine scheduler
AMP mixed precision
Gradient clipping
Label smoothing
CSV logging
structured checkpoint saving
```

如果机器没有 CUDA，AMP 会自动关闭，不会直接报错。

## 十四、正式结果表

等队员 A 跑完正式实验后，把结果填到这里。

| 模型 | 数据比例 | Best Val Acc | Best Val F1 | Best Epoch | 日志路径 | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CNN | 0.1 | 待填 | 待填 | 待填 | `logs/cnn_ratio0.1.csv` | `checkpoints/cnn_ratio0.1_best.pth` |
| CNN | 0.2 | 待填 | 待填 | 待填 | `logs/cnn_ratio0.2.csv` | `checkpoints/cnn_ratio0.2_best.pth` |
| CNN | 1.0 | 待填 | 待填 | 待填 | `logs/cnn_ratio1.0.csv` | `checkpoints/cnn_ratio1.0_best.pth` |
| ViT | 0.1 | 待填 | 待填 | 待填 | `logs/vit_ratio0.1.csv` | `checkpoints/vit_ratio0.1_best.pth` |
| ViT | 0.2 | 待填 | 待填 | 待填 | `logs/vit_ratio0.2.csv` | `checkpoints/vit_ratio0.2_best.pth` |
| ViT | 1.0 | 待填 | 待填 | 待填 | `logs/vit_ratio1.0.csv` | `checkpoints/vit_ratio1.0_best.pth` |

## 十五、常见问题

如果显存不够，可以降低 batch size：

```bash
python main.py --model vit --data_ratio 1.0 --epochs 100 --batch_size 64
```

如果只是检查代码，不要跑 100 epoch，使用：

```bash
python main.py --model vit --data_ratio 0.1 --epochs 2 --debug
```

如果要确认学习率是否正确，看终端输出或 CSV 里的 `lr` 列。

如果 checkpoint 加载失败，先确认使用的是对应模型的 checkpoint，比如 ViT 只能加载 `vit_ratio*_best.pth`。
