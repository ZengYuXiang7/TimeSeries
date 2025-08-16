import torch
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# 原始模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids=None, **kwargs):  # 兼容PEFT传入的多余参数
        x = input_ids
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        return self.linear3(x)


if __name__ == '__main__':
    model = SimpleModel()
    # 配置 LoRA
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        target_modules=["linear1", "linear2", "linear3"],  # 哪些层加 LoRA
        lora_dropout=0.1,
        bias="none",  # 是否对bias做lora（通常设为none）
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # 注入 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 简单模拟数据
    x = torch.randn(100, 128)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 训练1轮
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")