import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('WARNING: GPU is not found!')
    exit()

print(f'Using {device}')
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

# Run something with GPU
x = torch.rand(10).to(device)
x = x.sum().item()
print('Sum random numbers from GPU:', x)
