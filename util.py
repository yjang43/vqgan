
def compute_grad(parameters):
    total_norm = 0
    for p in parameters:
        if p is not None and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        else:
            print(p)
    total_norm = total_norm ** (1. / 2)
    return total_norm

def unnormalize(data, mean, std):
    data = data.permute(0, 2, 3, 1)
    data = data * std + mean
    return data.permute(0, 3, 1, 2)