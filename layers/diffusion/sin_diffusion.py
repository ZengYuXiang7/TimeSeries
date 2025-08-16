import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from layers.diffusion.blocks.noise_pred import NoisePredictor

len_seq, batch_size, n_steps, n_epochs = 32, 32, 1000, 500000
save_dir = f'results_step{n_steps}_epoch{n_epochs}'


# ========== Utilities ==========
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def vocab_embedding(vocabs, seq_len, device):
    vocab_to_idx = {"cos": 0, "sin": 1}
    indices = [vocab_to_idx[word] for word in vocabs]
    return torch.stack([
        torch.full((seq_len,), idx, dtype=torch.long, device=device) for idx in indices
    ])


# ========== Early Stopping ==========
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ========== Diffusion Process ==========
class GuidanceDenoiseDiffusion:
    def __init__(self, n_steps, device):
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar[t])[:, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, eps_model, x_t, t, vocab):
        alpha_t = self.alpha[t][:, None]
        alpha_bar_t = self.alpha_bar[t][:, None]
        eps_theta = eps_model(x_t, vocab, t)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - (1. - alpha_t) / torch.sqrt(1. - alpha_bar_t) * eps_theta)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.beta[t])[:, None]
            return mean + sigma_t * noise
        return mean

    def loss(self, eps_model, x_vocab):
        x_0 = torch.tensor([item[0] for item in x_vocab], dtype=torch.float32, device=self.device)
        vocabs = vocab_embedding([item[1] for item in x_vocab], x_0.shape[1], self.device)
        t = torch.randint(0, self.n_steps, (x_0.shape[0],), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, target = self.q_sample(x_0, t, noise)
        eps_theta = eps_model(x_t, vocabs, t)
        return torch.nn.functional.mse_loss(eps_theta, target)


# ========== Data Generation ==========
def generate_data(batch_size, seq_len):
    x = np.linspace(0, 2 * np.pi, seq_len)
    return [(np.sin(x), 'sin') if i % 2 == 0 else (np.cos(x), 'cos') for i in range(batch_size)]

def generate_noisy_wave(batch_size, seq_len, noise_scale, mode):
    x = np.linspace(0, 2 * np.pi, seq_len)
    if mode == 'sin':
        return torch.tensor([np.sin(x) + np.random.normal(0, noise_scale, seq_len) for _ in range(batch_size)], dtype=torch.float32)
    return torch.tensor([np.cos(x) + np.random.normal(0, noise_scale, seq_len) for _ in range(batch_size)], dtype=torch.float32)


# ========== Visualization ==========
def plot_sample(epoch, conditional, seq_len, n_steps, diffusion, model, device, save_path):
    x_input = generate_noisy_wave(1, seq_len, 0.3, conditional).to(device)
    target = np.sin(np.linspace(0, 2 * np.pi, seq_len)) if conditional == 'sin' else np.cos(np.linspace(0, 2 * np.pi, seq_len))
    for t in range(n_steps - 1, -1, -1):
        t_tensor = torch.full((1,), t, device=device)
        x_input = diffusion.p_sample(model, x_input, t_tensor, vocab_embedding([conditional], seq_len, device))
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 2 * np.pi, seq_len), target, '--', label=f'True {conditional}(x)')
    plt.plot(np.linspace(0, 2 * np.pi, seq_len), x_input[0].cpu().numpy(), label='Generated')
    plt.legend()
    plt.title(f'{conditional} - Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/sample_epoch_{conditional}_{epoch + 1}.png')
    plt.close()


def eval_model(model_path, seq_len=32, n_steps=1000, conditional='cos'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和扩散器
    model = NoisePredictor(seq_len, 2, device).to(device)
    diffusion = GuidanceDenoiseDiffusion(n_steps=n_steps, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 初始纯噪声
    x_input = torch.randn(1, seq_len).to(device)

    with torch.no_grad():
        for t in range(n_steps - 1, -1, -1):
            t_tensor = torch.full((1,), t, device=device)
            vocab = vocab_embedding([conditional], seq_len, device)
            x_input = diffusion.p_sample(model, x_input, t_tensor, vocab)

    # 绘图对比
    x = np.linspace(0, 2 * np.pi, seq_len)
    ground_truth = np.cos(x) if conditional == 'cos' else np.sin(x)

    plt.figure(figsize=(10, 4))
    plt.plot(x, ground_truth, '--', label=f'True {conditional}(x)')
    plt.plot(x, x_input[0].cpu().numpy(), label='Generated')
    plt.legend()
    plt.title(f'Test Output - {conditional}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_epoch_{conditional}_{epoch + 1}.png')

# ========== Main Training ==========
if __name__ == "__main__":
    setup_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f'{save_dir}/sin', exist_ok=True)
    os.makedirs(f'{save_dir}/cos', exist_ok=True)

    eps_model = NoisePredictor(len_seq, 2, device).to(device)
    diffusion = GuidanceDenoiseDiffusion(n_steps, device)

    optimizer = torch.optim.Adam(eps_model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=20000, delta=0.0, verbose=False)

    for epoch in trange(n_epochs):
        x_vocab = generate_data(batch_size, len_seq)
        optimizer.zero_grad()
        loss = diffusion.loss(eps_model, x_vocab)
        loss.backward()
        optimizer.step()

        early_stopping(loss.item())
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            with torch.no_grad():
                plot_sample(epoch, 'sin', len_seq, n_steps, diffusion, eps_model, device, f'{save_dir}/sin')
                plot_sample(epoch, 'cos', len_seq, n_steps, diffusion, eps_model, device, f'{save_dir}/cos')

    torch.save(eps_model.state_dict(), f'{save_dir}/model.pt')
    print("Training completed.")

    eval_model('./results_step1000_epoch500000/model.pt', conditional='sin')
    eval_model('./results_step1000_epoch500000/model.pt', conditional='cos')