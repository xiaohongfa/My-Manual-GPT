# My Manual GPT

手写的 decoder-only Transformer，保留模型实现本身。

包含 RMSNorm、RoPE、因果自注意力、SwiGLU、残差连接、交叉熵损失及贪心生成。

默认配置：4 层、宽度 64、4 个注意力头、前馈宽度 192、词表大小 8、上下文长度 151，共 214,592 个可训练参数。所有配置可通过 `GPTConfig` 修改。

## 使用

依赖 Python 3.10+ 和 PyTorch。安装适合设备的 PyTorch 后，可直接导入：

```python
import torch
from model import GPT, GPTConfig

model = GPT(GPTConfig())
x = torch.tensor([[1, 4, 5, 4]])
y = torch.tensor([[4, 5, 4, 2]])  # 与 x 对齐的下一 token 标签

logits, loss = model(x, y)
loss.backward()
print(logits.shape, loss.item())

output = model.generate(
    prompt=[1, 4, 5, 2],
    eos_id=3,
    max_new_tokens=16,
    device=torch.device("cpu"),
)
print(output)
```

输入是 token ID；分词、数据准备和训练循环由调用方提供。损失标签中的 `-100` 会被忽略。生成返回新增的 token ID，遇到 EOS 即停止。

仓库仅含源码；新建模型为随机初始化，不含训练权重。
