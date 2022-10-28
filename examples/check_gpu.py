"""Check if GPU is available. """
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')

        print(f'Using: {device}')
        print('GPU Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
else:
    device = torch.device('cpu')
    print('WARNING: GPU is not found!')
    print(f'Running on Device: {device}')
    exit()

# Run a trivial task
x = torch.rand(10).to(device)
x = x.sum().item()
print(f'Sum random numbers from {device}:', x)
