import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, codebook):
        z_sq = torch.sum(z**2, dim=1, keepdim=True)
        codebook_sq = torch.sum(codebook**2, dim=1)
        sq = z_sq + codebook_sq

        norm = torch.addmm(sq, z, codebook.T, beta=1, alpha=-2)
        indices = norm.argmin(dim=1)

        z_q = codebook.index_select(0, indices)

        ctx.save_for_backward(codebook, indices)
        ctx.mark_non_differentiable(indices)
        return z_q
    
    @staticmethod
    def backward(ctx, z_q_grad):
        codebook, indices = ctx.saved_tensors
        z_grad, codebook_grad = None, None

        if ctx.needs_input_grad[0]:
            z_grad = z_q_grad
        
        if ctx.needs_input_grad[1]:
            codebook_grad = torch.zeros_like(codebook)
            codebook_grad[indices] = z_q_grad
        
        return z_grad, codebook_grad


# class VectorQuantizer(nn.Module):
#     def __init__(self):
#         super(VectorQuantizer, self).__init__()
        
#         self._embedding = nn.Parameter(torch.rand(512, 256))

#     def forward(self, inputs):
        
#         # Calculate distances
#         distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
#                     + torch.sum(self._embedding**2, dim=1)
#                     - 2 * torch.matmul(inputs, self._embedding.T))
            
#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1)
#         z_q = self._embedding.index_select(0, encoding_indices)
        
        
#         # Loss
#         rec_z_q = inputs + (z_q - inputs).detach()
        
#         # convert quantized from BHWC -> BCHW
#         return rec_z_q, z_q