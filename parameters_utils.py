import torch

def count_parameters(model):
    total_params = 0
    total_zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total_params += module.weight.numel()
            zero_count = torch.sum(module.weight == 0)
            total_zero_params += zero_count
            sparsity = 100.0 * float(zero_count) / float(module.weight.nelement())
            print(f'Sparsity in {name}.weight with {module.weight.numel()} parameters: {sparsity:.3f}%')
    global_sparsity = 100.0 * float(total_zero_params) / float(total_params)
    print(f'Global sparsity: {global_sparsity:.3f}%')
    return total_params
