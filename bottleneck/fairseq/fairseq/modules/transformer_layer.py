# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import math
import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import einops
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from bitsandbytes.optim import GlobalOptimManager
from heapq import heappush, heappop, heapify

def get_compression(x:torch.Tensor)->float:
    """Yields the compression rate of Huffman Coding"""
    assert x.device.type == 'cuda'
    assert x.dtype in [torch.float32, torch.float16]

    C, S = bnb.functional.quantize_blockwise(x)
    val, counts = torch.unique(C.int(), return_counts=True)

    symb2freq = {}
    for i, (c, count) in enumerate(zip(val, counts)):
        symb2freq[c.item()] = count.item()

    huff = encode(symb2freq)
    total_bits = 0
    for p in huff:
        total_bits += len(p[1])*symb2freq[p[0]]

    return 1.0-(total_bits/(C.numel()*8))

# taken from: https://rosettacode.org/wiki/Huffman_coding#Python
def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

class Comm8bitFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        #print(get_compression(x), 'fw')
        C, S = bnb.functional.quantize_blockwise(x)
        output = bnb.functional.dequantize_blockwise(C, S)
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        #print(get_compression(grad_output), 'bw')
        C, S = bnb.functional.quantize_blockwise(grad_output)
        grad_input = bnb.functional.dequantize_blockwise(C, S)

        return grad_input.to(grad_output.dtype)


class LinearFunction(torch.autograd.Function):

    @staticmethod
    def get_8bit_linear_trimmed(x, stochastic=False, trim_value=3.0):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        norm = math.sqrt(math.pi)/math.sqrt(2.0)
        #std = torch.abs(x).mean()*norm
        std = torch.std(x)
        max1 = std*trim_value
        x = x/max1*127
        x = round_func(x)
        x[x > 127] = 127
        x[x < -127] = -127
        x = x/127*max1

        return x

    def quant(x, quant_type, dim=1):
        if quant_type == 'linear':
            max1 = torch.abs(x).max().float()
            xq = torch.round(x/max1*127).to(torch.int8)
            return xq, max1
        elif quant_type == 'vector':
            max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
            xq = torch.round(x/max1*127).to(torch.int8)
            return xq, max1
        elif quant_type == 'min-max':
            maxA = torch.amax(x, dim=dim, keepdim=True).float()
            minA = torch.amin(x, dim=dim, keepdim=True).float()
            scale = (maxA-minA)/2.0
            xq = torch.round(127*(x-minA-scale)/scale).to(torch.int8)
            return xq, (minA.float(), scale.float())
        else: return None

    def dequant(xq, S1, S2, dtype, quant_type):
        if quant_type == 'linear':
            norm = S1*S2/(127*127)
            # double cast needed to prevent overflows
            return (xq.float()*norm).to(dtype)
        elif quant_type == 'vector':
            x = xq.float()
            if len(xq.shape) == 2 and len(S1.shape) == 3: S1 = S1.squeeze(0)
            if len(xq.shape) == 2 and len(S2.shape) == 3: S2 = S2.squeeze(0)
            #print(x.shape, S1.shape, S2.shape)
            if len(S1.shape) == 2:
                x *= S1.t()/127
            else:
                x *= S1/127
            x *= S2/127
            return x.to(dtype)
        else: return None

    def dequant_min_max(xq, A, B, SA, SB, dtype):
        offset = B.float().t().sum(0)*(SA[0]+SA[1])
        x = xq.float()
        if len(xq.shape) == 2 and len(SB.shape) == 3: SB = SB.squeeze(0)
        if len(xq.shape) == 2 and len(SA.shape) == 3: SA = SA.squeeze(0)
        if len(SB.shape) == 2:
            x *= SB.t()/127
        else:
            x *= SB/127
        x *= SA[1]/127
        x +=offset
        return x.to(dtype)


    def get_8bit_linear(x, stochastic=False):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        max1 = torch.abs(x).max()
        x = x/max1*127
        x = round_func(x)/127*max1
        #x = torch.round(x)/128*max1
        return x

    @staticmethod
    def get_8bit_vector_wise(x, dim, stochastic=False):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        max1[max1==0] = 1.0
        x = (x*127)/max1
        x = round_func(x)/127*max1
        return x

    @staticmethod
    def round_stoachastic(x):
        sign = torch.sign(x)
        absx = torch.abs(x)
        decimal = absx-torch.floor(absx)
        rdm = torch.rand_like(decimal)
        return sign*(torch.floor(absx)+(rdm < decimal).to(x.dtype))

    @staticmethod
    def fake_8bit_storage(w, exponent_bits):
        code = bnb.functional.create_dynamic_map(n=exponent_bits).to(w.device)
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_quantile(w, args):
        code = bnb.functional.estimate_quantiles(w.data, offset=args.offset)
        #C = bnb.functional.quantize_no_absmax(code, w)
        #out = bnb.functional.dequantize_no_absmax(code, C, out=w.data)
        #print(out)
        #out = out.half()
        code /= torch.max(torch.abs(code))
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_stoachstic(w):
        rand = torch.rand(1024, device=w.device)
        absmax, C = bnb.functional.quantize_blockwise(w.data, rand=rand)
        out = bnb.functional.dequantize_blockwise(absmax, C)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_with_max(w, topk=8):
        blocked_w = einops.rearrange(w.flatten(), '(h b) -> h b', b=256)
        max_val, idx = torch.sort(torch.abs(blocked_w), dim=1, descending=True)
        idx = idx[:, :topk]
        max_val = max_val[:, :topk]

        mask = torch.zeros_like(blocked_w)
        mask.scatter_(dim=1, index=idx, src=torch.ones_like(max_val))
        mask = mask.bool()

        # 1. zero out max values
        # 2. quantize + dequantize
        # 3. write back max values
        # 4. copy matrix back to weight

        values = blocked_w[mask]
        blocked_w[mask] = 0

        code = bnb.functional.create_dynamic_map()
        code = code.to(w.device)
        absmax, C = bnb.functional.quantize_blockwise(blocked_w.data)
        bnb.functional.dequantize_blockwise(absmax, C, out=blocked_w)

        blocked_w[mask] = values

        unblocked_w = blocked_w.flatten().view(w.shape)

        w.copy_(unblocked_w)
        return unblocked_w


    @staticmethod
    def forward(ctx, x, weight, bias=None, args=None):
        if args.use_8bit_training != 'off':
            weight8, S1 = LinearFunction.quant(weight, args.quant_type, dim=1)
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=2)
            outputq = bnb.functional.igemm(x8, weight8.t())
            output = LinearFunction.dequant(outputq, S1, S2, x.dtype, args.quant_type)
            #if torch.rand(1) < 0.01:
                #output32 = torch.matmul(x, weight.t())
                #err = torch.abs(output-output32).float()
                #relerr = err/(torch.abs(output32).float()+1e-8)
                #print(f'{err.mean().item():.4f}, {relerr.mean().item():.4f}', args.quant_type, 'forward', proxy)
        else:
            #output = torch.matmul(x, weight.t())
            output = torch.einsum('bsi,oi->bso', x, weight)

        ctx.save_for_backward(x, weight, bias)
        ctx.args = args

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        args = ctx.args
        stochastic = False
        grad_input = grad_weight = grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]: grad_bias = grad_output.sum(0)

        # weight and x are already 8bit
        # -> transform grad_output to 8-bit
        if args.use_8bit_training == 'forward+wgrad':
            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=[0, 1])
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=[0, 1])
            grad_weight8 = bnb.functional.igemm(grad_output8, x8)
            grad_weight = LinearFunction.dequant(grad_weight8, S1, S2, grad_output.dtype, args.quant_type)

            #grad_weight32 = torch.einsum('bso,bsi->oi', grad_output, x)

            grad_input = grad_output.matmul(weight)
        elif args.use_8bit_training == 'full':
            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=[0, 1])
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=[0, 1])
            grad_weight8 = torch.zeros_like(weight, dtype=torch.int32)
            bnb.functional.igemm(grad_output8, x8, out=grad_weight8)
            grad_weight = LinearFunction.dequant(grad_weight8, S1, S2, grad_output.dtype, args.quant_type)

            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=2)
            weight8, S3 = LinearFunction.quant(weight, args.quant_type, dim=0)
            grad_input8 = bnb.functional.igemm(grad_output8, weight8)
            grad_input = LinearFunction.dequant(grad_input8, S1, S3, grad_output.dtype, args.quant_type)

            #if torch.rand(1) < 0.01:
            #    grad_weight32 = torch.einsum('bso,bsi->oi', grad_output, x)
            #    grad_input32 = torch.einsum('bso, oi->bsi', grad_output, weight)
            #    err = torch.abs(grad_weight-grad_weight32).float()
            #    relerr = err/(torch.abs(grad_weight32).float()+1e-8)
            #    print(f'{err.mean().item():.4f}, {relerr.mean().item():.4f}', args.quant_type, 'weight grad')
            #    err = torch.abs(grad_input-grad_input32).float()
            #    relerr = err/(torch.abs(grad_input32).float()+1e-8)
            #    print(f'{err.mean().item():.4f}, {relerr.mean().item():.4f}', args.quant_type, 'delta')

        else:
            grad_input = grad_output.matmul(weight)
            grad_weight = torch.einsum('bsi,bso->oi', x, grad_output)

        return grad_input, grad_weight, grad_bias, None

class Linear8bit(nn.Module):
    def __init__(self, input_features, output_features, bias=True, args=None):
        super(Linear8bit, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.args = args

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        self.args.training = self.training

        return LinearFunction.apply(x, self.weight, self.bias, self.args)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

class SparseFF_Block(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(SparseFF_Block, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        if args.str_type == 'scalar':
            #self.s1 = nn.Parameter(torch.rand(1) + torch.ones(1)*args.str_init)
            #self.s2 = nn.Parameter(torch.rand(1) + torch.ones(1)*args.str_init)
            self.s1 = nn.Parameter(torch.ones(1)*args.str_init)
            self.s2 = nn.Parameter(torch.ones(1)*args.str_init)
        elif args.str_type == 'vector':
            self.s1 = nn.Parameter(torch.rand(dim2, 1) + torch.ones(dim2, 1)*args.str_init)
            self.s2 = nn.Parameter(torch.rand(1, dim2) + torch.ones(1, dim2)*args.str_init)
        elif args.str_type == 'matrix':
            self.s1 = nn.Parameter(torch.rand(dim2, dim1) + torch.ones(dim2, dim1)*args.str_init)
            self.s2 = nn.Parameter(torch.rand(dim1, dim2) + torch.ones(dim1, dim2)*args.str_init)
        else:
            raise ValueError(f'Sparse type not supported: {args.str_type}')
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        if args.str_sparse:
            GlobalOptimManager.get_instance().override_config([self.fc1.weight, self.fc2.weight], 'is_sparse', True)
        self.register_buffer('iter', torch.zeros(1, dtype=torch.long))
        self.progress = 0.0
        self.args = args
        self.oversparse = False
        self.lossw = args.str_loss
        self.init = False
        if args.str_lr > 0.0:
            GlobalOptimManager.get_instance().override_config([self.s1, self.s2], 'lr' , args.str_lr)

    def forward(self, x):
        if self.training:
            self.iter += 1
            self.progress = self.get_expected_sparsity()[0]

        if not self.init:
            self.init = True
            with torch.no_grad():
                self.s1[:] = self.args.str_init
                self.s2[:] = self.args.str_init

        if self.progress <= self.args.str_progress_start:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        else:
            if self.args.str_noisy_relu == 0.0 and self.args.str_weight_noise == 0.0:
                w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-torch.sigmoid(self.s1))
                w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-torch.sigmoid(self.s2))
            elif self.args.str_weight_noise > 0.0:
                if self.training:
                    w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-torch.sigmoid(self.s1))
                    w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-torch.sigmoid(self.s2))
                    mask1 = torch.rand_like(w1) < self.args.str_weight_noise
                    mask2 = torch.rand_like(w2) < self.args.str_weight_noise
                    w1 = self.fc1.weight*mask1.to(w1.dtype) + (w1*(mask1==0).to(w1.dtype))
                    w2 = self.fc2.weight*mask2.to(w1.dtype) + (w2*(mask2==0).to(w1.dtype))
                else:
                    w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-torch.sigmoid(self.s1))
                    w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-torch.sigmoid(self.s2))
            elif self.args.str_noisy_relu > 0.0:
                noise_threshold = 1.0-self.progress
                if self.training:
                    w1 = torch.abs(self.fc1.weight)-torch.sigmoid(self.s1)
                    w2 = torch.abs(self.fc2.weight)-torch.sigmoid(self.s2)
                    max1 = torch.max(w1)
                    max2 = torch.max(w2)
                    w1 = torch.sign(self.fc1.weight)*torch.relu(w1+((torch.rand_like(w1)-0.5)*noise_threshold*self.args.str_noisy_relu*max1))
                    w2 = torch.sign(self.fc2.weight)*torch.relu(w2+((torch.rand_like(w2)-0.5)*noise_threshold*self.args.str_noisy_relu*max2))
                else:
                    w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-torch.sigmoid(self.s1))
                    w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-torch.sigmoid(self.s2))

            x = self.activation_fn(F.linear(x, w1, self.fc1.bias))
            x = self.activation_dropout_module(x)
            x = F.linear(x, w2, self.fc2.bias)

            if self.training and self.iter % 50 == 0:
                sp1 = self.get_sparsity(w1)
                sp2 = self.get_sparsity(w2)
                spE = self.get_expected_sparsity()[1]
                sp = (sp1+sp2)/2.0
                if sp >= self.args.target_sparsity:
                    self.lossw = 0.0
                elif sp > spE:
                    self.oversparse = True
                    self.lossw *= 1.0 - self.args.str_mult_offset
                else:
                    self.oversparse = False
                    self.lossw *= 1.0 + self.args.str_mult_offset

            if (self.iter+1) % (50) == 0:
                sp1 = self.get_sparsity(w1)
                sp2 = self.get_sparsity(w2)
                spE = self.get_expected_sparsity()[1]
                sp = (sp1+sp2)/2.0
                #s1, s2 = self.get_expected_sparsity_thresholds()
                #self.s1.data[:] = s1
                #self.s2.data[:] = s2
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print('Sparsity: {0:.4f}, expected: {1:.4f}'.format((sp1+sp2)*0.5, spE), self.s1.mean().item(), self.lossw, self.iter.item())

        return x

    def calc_loss(self):
        if not self.training: return 0.0
        if self.progress < self.args.str_progress_start: return 0.0
        #if self.oversparse and self.args.str_oversparse: return 0.0
        if self.args.str_loss > 0.0:
            loss = ((self.s2.float()**2).sum()+(self.s1.float()**2).sum())*self.lossw
            return loss
        else:
            return 0.0

    @torch.no_grad()
    def get_expected_sparsity_thresholds(self):
        sparsity = self.get_expected_sparsity()[1]
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        vals, idx = torch.sort(w1.flatten())
        n = w1.numel()
        thresh1 = torch.logit(vals[int(sparsity*n)])

        vals, idx = torch.sort(w2.flatten())
        n = w2.numel()
        thresh2 = torch.logit(vals[int(sparsity*n)])

        return thresh1, thresh2

    def get_sparsity(self, w):
        return (w.detach().data==0.0).sum()/w.numel()

    @torch.no_grad()
    def get_expected_sparsity(self):
        current_step = self.iter.item() / self.args.update_freq[0]
        progress = current_step / self.args.max_update
        sparse_start_step = self.args.max_update*self.args.str_progress_start
        offset_progress = (current_step - sparse_start_step)/(self.args.max_update-sparse_start_step)
        if offset_progress < 0.0: offset_progress = 0.0
        return progress, self.args.target_sparsity*offset_progress

class NonLinearFF(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(NonLinearFF, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        if args.str_type == 'scalar':
            self.s1 = nn.Parameter(torch.ones(1)*args.str_init)
            self.s2 = nn.Parameter(torch.ones(1)*args.str_init)
        elif args.str_type == 'vector':
            self.s1 = nn.Parameter(torch.ones(dim2, 1)*args.str_init)
            self.s2 = nn.Parameter(torch.ones(1, dim2)*args.str_init)
        else:
            raise ValueError(f'Sparse type not supported: {args.str_type}')
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)

        if args.str_sparse:
            GlobalOptimManager.get_instance().override_config([self.fc1.weight, self.fc2.weight], 'is_sparse', True)
        self.register_buffer('iter', torch.zeros(1, dtype=torch.long))
        self.progress = 0.0
        self.args = args
        if args.str_func == 'logistic':
            self.func = torch.sigmoid
        elif args.str_func == 'linear':
            self.func = lambda x: x
        elif args.str_func == 'tanh':
            self.func = torch.tanh
        else:
            raise ValueError(f'function not supported: {args.str_func}!')

    def forward(self, x):
        if self.training:
            self.iter += 1
            self.progress = self.get_expected_sparsity()[0]

        if self.args.str_noisy_relu == 0.0:
            w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-self.func(self.s1))
            w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-self.func(self.s2))
        elif self.args.str_noisy_relu > 0.0:
            noise_threshold = 1.0-self.progress
            if self.training:
                w1 = torch.abs(self.fc1.weight)-self.func(self.s1)
                w2 = torch.abs(self.fc2.weight)-self.func(self.s2)
                max1 = torch.max(w1)
                max2 = torch.max(w2)
                w1 = torch.sign(self.fc1.weight)*torch.relu(w1+((torch.rand_like(w1)-0.5)*noise_threshold*self.args.str_noisy_relu*max1))
                w2 = torch.sign(self.fc2.weight)*torch.relu(w2+((torch.rand_like(w2)-0.5)*noise_threshold*self.args.str_noisy_relu*max2))
            else:
                w1 = torch.sign(self.fc1.weight)*torch.relu(torch.abs(self.fc1.weight)-self.func(self.s1))
                w2 = torch.sign(self.fc2.weight)*torch.relu(torch.abs(self.fc2.weight)-self.func(self.s2))

        x = self.activation_fn(F.linear(x, w1, self.fc1.bias))
        x = self.activation_dropout_module(x)
        x = F.linear(x, w2, self.fc2.bias)

        if (self.iter+1) % (1000) == 0:
            sp1 = self.get_sparsity(w1)
            sp2 = self.get_sparsity(w2)
            spE = self.get_expected_sparsity()[1]
            sp = (sp1+sp2)/2.0
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Sparsity: {0:.4f}, expected: {1:.4f}'.format((sp1+sp2)*0.5, spE), self.s1.mean().item(), self.iter.item())

        return x

    def get_sparsity(self, w):
        return (w.detach().data==0.0).sum()/w.numel()

    @torch.no_grad()
    def get_expected_sparsity(self):
        current_step = self.iter.item() / self.args.update_freq[0]
        progress = current_step / self.args.max_update
        sparse_start_step = self.args.max_update*self.args.str_progress_start
        offset_progress = (current_step - sparse_start_step)/(self.args.max_update-sparse_start_step)
        if offset_progress < 0.0: offset_progress = 0.0
        return progress, self.args.target_sparsity*offset_progress

class FF_Block(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(FF_Block, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)

        if args.init == 'pytorch':
            # usual init
            pass
        elif args.init == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)
        elif args.init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fc1.weight)
            torch.nn.init.kaiming_normal_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)
        elif args.init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

class FFBottleneck(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(FFBottleneck, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        assert not (args.scale_factor > 1 and args.maxout > 1)
        if args.scale_factor > 1:
            self.fc1 = nn.Linear(dim1, dim1//args.scale_factor)
            self.fc2 = nn.Linear(dim1//args.scale_factor, dim1)
            self.norm1 = nn.LayerNorm(dim1)
            self.norm2 = nn.LayerNorm(dim1//args.scale_factor)
            #self.norm3 = nn.LayerNorm(dim1)
        elif args.maxout > 1:
            self.fc2 = nn.Linear(dim1//args.maxout, dim1)
            self.norm1 = nn.LayerNorm(dim1)
            self.norm2 = nn.LayerNorm(dim1//args.maxout)
        else:
            raise ValueError(f'invalid bottleneck function')

        self.args = args

    def forward(self, x):
        if self.args.maxout > 1:
            #x = x.float()
            x = self.norm1(x)
            x = einops.rearrange(x, 's b (h i)-> s b h i', i=self.args.maxout)
            x = torch.max(x, -1, keepdim=False).values
            x = self.norm2(x)
            #x = x.half()
            if self.args.comm8bit:
                x = Comm8bitFunction.apply(x)
            x = self.fc2(x)
        else:
            #x = F.relu(x, inplace=True)
            #x = x.float()
            x = self.norm1(x)
            x = self.fc1(x)
            x = self.norm2(x)
            #x = x.half()
            if self.args.comm8bit:
                x = Comm8bitFunction.apply(x)
            x = self.fc2(x)
            #x = self.norm3(x)

        return x



class DendriticLinear(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(DendriticLinear, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        self.args = args
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.iter = 0

        self.g1a = nn.Linear(dim1, args.dim3)
        self.g1b = nn.Linear(args.dim3, dim1)
        self.g2a = nn.Linear(dim2, args.dim3)
        self.g2b = nn.Linear(args.dim3, dim2)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.g1a.weight)
            nn.init.xavier_uniform_(self.g1b.weight)
            nn.init.xavier_uniform_(self.g2a.weight)
            nn.init.xavier_uniform_(self.g2b.weight)
            nn.init.zeros_(self.g1a.bias)
            nn.init.zeros_(self.g1b.bias)
            nn.init.zeros_(self.g2a.bias)
            nn.init.zeros_(self.g2b.bias)

    def calc_loss(self):
        #return (self.top1_freq*self.mean_p).mean()*self.args.dloss*1e3
        return 0.0

    def forward(self, x):
        self.iter += 1

        if self.args.dendrite_type in ['first', 'both']:
            blocked_p1 = torch.softmax(einops.rearrange(self.g1b(self.g1a(x)), 'b s (h n) -> b s h n', n=self.args.branch_size), dim=-1)
            maxval1, idx1 = torch.sort(blocked_p1, dim=-1, descending=True)
            maxval1[:, :, :, self.args.topk:] = 0.0
            mask1 = torch.zeros_like(blocked_p1)
            mask1.scatter_(dim=-1, index=idx1, src=maxval1)
            x = x*mask1.flatten().view(x.shape)

        if self.args.no_relu:
            x = self.fc1(x)
        else:
            x = self.activation_fn(self.fc1(x))

        if self.args.dendrite_type in ['second', 'both']:
            blocked_p2 = torch.softmax(einops.rearrange(self.g2b(self.g2a(x)), 'b s (h n) -> b s h n', n=self.args.branch_size), dim=-1)
            maxval2, idx2 = torch.sort(blocked_p2, dim=-1, descending=True)
            maxval2[:, :, :, self.args.topk:] = 0.0
            mask2 = torch.zeros_like(blocked_p2)
            mask2.scatter_(dim=-1, index=idx2, src=maxval2)
            x = x*mask2.flatten().view(x.shape)

        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        return x


class Linear8bitBlock(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(Linear8bitBlock, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        self.fc1 = Linear8bit(dim1, dim2)
        self.fc2 = Linear8bit(dim2, dim1)

        if args.init == 'pytorch':
            # usual init
            pass
        elif args.init == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)
        elif args.init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fc1.weight)
            torch.nn.init.kaiming_normal_(self.fc2.weight)
        elif args.init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

class L1FF_Block(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(L1FF_Block, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.Activation_fn = func
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.args = args
        self.iters = 0

    def forward(self, x):
        self.iters += 1
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        if (self.iters / self.args.update_freq[0]) % 250 == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Sparsity: {0:.4f}'.format((self.get_sparsity(self.fc1.weight)+self.get_sparsity(self.fc2.weight))/2.0))
        return x

    def get_sparsity(self, w):
        return (w.detach().data==0.0).sum()/w.numel()

    def calc_loss(self):
        if not self.training: return 0.0
        return self.args.l1*(torch.abs(self.fc1.weight.float()).sum()+torch.abs(self.fc2.weight.float()).sum())


class RdmProjectionFF(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(RdmProjectionFF, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        self.register_parameter('bias1', nn.Parameter(torch.zeros(dim2)))
        self.register_parameter('bias2', nn.Parameter(torch.zeros(dim1)))
        print('generating rdm numbers...')
        if args.rdm_type == 'full':
            self.register_parameter('fc1', nn.Parameter(torch.zeros(dim1, 128)))
            self.register_parameter('fc2', nn.Parameter(torch.zeros(dim2, 128)))
            self.register_buffer('rdm1', torch.randn(128, dim1, dim2)*0.1)
            self.register_buffer('rdm2', torch.randn(128, dim2, dim1)*0.1)
        elif args.rdm_type == 'half':
            self.register_parameter('fc1', nn.Parameter(torch.zeros(dim1, 128)))
            self.register_parameter('fc2', nn.Parameter(torch.zeros(dim2, 128)))
            self.register_buffer('rdm1', torch.randn(128, dim2)*0.1)
            self.register_buffer('rdm2', torch.randn(128, dim1)*0.1)
        elif args.rdm_type == 'halfw':
            self.register_parameter('fc1', nn.Parameter(torch.zeros(dim1, 128)))
            self.register_parameter('fc2', nn.Parameter(torch.zeros(dim2, 128)))
            self.register_parameter('w1', nn.Parameter(torch.zeros(dim2, 1)))
            self.register_parameter('w2', nn.Parameter(torch.zeros(dim1, 1)))
            self.register_buffer('rdm1', torch.randn(128, dim2)*0.1)
            self.register_buffer('rdm2', torch.randn(128, dim1)*0.1)
            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.w2)
        elif args.rdm_type == 'double_nonlinear':
            k1, k2 = dim1//16, dim2//16
            self.register_parameter('fc1', nn.Parameter(torch.zeros(dim1, 128)))
            self.register_parameter('fc2', nn.Parameter(torch.zeros(dim2, 128)))
            self.register_buffer('rdm1', torch.randn(128, k1)*0.1)
            self.register_buffer('rdm2', torch.randn(128, k2)*0.1)
            self.register_parameter('fc1b', nn.Parameter(torch.zeros(k1, dim2)))
            self.register_parameter('fc2b', nn.Parameter(torch.zeros(k2, dim1)))
        elif args.rdm_type == 'linearcomb':
            self.register_parameter('fc1', nn.Parameter(torch.zeros(dim1, 128)))
            self.register_parameter('fc2', nn.Parameter(torch.zeros(dim2, 128)))
            self.register_buffer('rdm1', torch.randn(128, dim2, 16)*0.1)
            self.register_buffer('rdm2', torch.randn(128, dim1, 16)*0.1)
            self.register_parameter('fc1b', nn.Parameter(torch.zeros(16)))
            self.register_parameter('fc2b', nn.Parameter(torch.zeros(16)))
            #self.prelu1 = nn.ELU(inplace=True)
            #self.prelu2 = nn.ELU(inplace=True)
            self.prelu1 = nn.PReLU(dim2)
            self.prelu2 = nn.PReLU(dim1)
            with torch.no_grad():
                nn.init.uniform_(self.fc1b)
                nn.init.uniform_(self.fc2b)

        self.norm1 = nn.LayerNorm(dim2)
        self.norm2 = nn.LayerNorm(dim1)
        self.args = args

        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1)
            nn.init.xavier_uniform_(self.fc2)
            nn.init.xavier_uniform_(self.rdm1)
            nn.init.xavier_uniform_(self.rdm2)


    def forward(self, x):
        if self.args.rdm_type == 'full':
            w1 = torch.einsum('hr,rho->oh', self.fc1, self.rdm1)
            w2 = torch.einsum('hr,rho->oh', self.fc2, self.rdm2)
        elif self.args.rdm_type == 'half':
            w1 = torch.einsum('hr,ro->oh', self.fc1, self.rdm1)
            w2 = torch.einsum('hr,ro->oh', self.fc2, self.rdm2)
        elif self.args.rdm_type == 'halfw':
            w1 = torch.einsum('hr,ro->oh', self.fc1, self.rdm1)*self.w1
            w2 = torch.einsum('hr,ro->oh', self.fc2, self.rdm2)*self.w2
        elif self.args.rdm_type == 'double_nonlinear':
            w1 = torch.relu(torch.einsum('hr,rk->kh', self.fc1, self.rdm1))
            w2 = torch.relu(torch.einsum('hr,rk->kh', self.fc2, self.rdm2))
            w1 = torch.einsum('kh,ko->oh', w1, self.fc1b)
            w2 = torch.einsum('kh,ko->oh', w2, self.fc2b)
        elif self.args.rdm_type == 'linearcomb':
            o1 = self.prelu1(torch.einsum('hr,roi->ohi', self.fc1, self.rdm1).unsqueeze(0)).squeeze(0)
            o2 = self.prelu2(torch.einsum('hr,roi->ohi', self.fc2, self.rdm2).unsqueeze(0)).squeeze(0)
            #o1 = torch.einsum('hr,roi->ohi', self.fc1, self.rdm1a)
            #o2 = torch.einsum('hr,roi->ohi', self.fc2, self.rdm2a)
            w1 = (o1*self.fc1b.view(1, 1, -1)).sum(-1)
            w2 = (o2*self.fc2b.view(1, 1, -1)).sum(-1)

        x = self.activation_fn(self.norm1(F.linear(x, w1, self.bias1)))
        x = F.linear(x, w2, self.bias2)
        x = self.norm2(x)

        return x

class FixedSparseFF(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(FixedSparseFF, self).__init__()
        self.activation_dropout_module = FairseqDropout(float(activation_drop), module_name=self.__class__.__name__ )
        self.activation_fn = func
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.mask1 = None
        self.mask2 = None
        self.args = args

    def forward(self, x):
        if self.mask1 is None:
            self.mask1 = torch.rand(self.fc1.weight.shape, device=x.device) > self.args.target_sparsity
            self.mask2 = torch.rand(self.fc2.weight.shape, device=x.device) > self.args.target_sparsity
        self.fc1.weight.data[self.mask1] = 0.0
        self.fc2.weight.data[self.mask2] = 0.0
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

class LinearFF_Block(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(LinearFF_Block, self).__init__()
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class WeightDropFF(nn.Module):
    def __init__(self, args, dim1, dim2, func=torch.relu, activation_drop=0.0):
        super(WeightDropFF, self).__init__()
        self.activation_fn = func
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, dim1)

    @torch.no_grad()
    def get_mask(self, w):
        r1 = torch.rand(w.numel(), device=w.device)
        r1 = r1.view(r1.numel()//4, 4)
        idx = torch.argsort(r1, axis=1)
        idx[idx< 2] = 0
        idx[idx>= 2] = 1
        idx = idx.bool()
        idx = idx.view(w.shape)
        return idx

    def forward(self, x):
        if self.training:
            w1 = self.fc1.weight
            w2 = self.fc2.weight
            m1 = self.get_mask(w1)
            m2 = self.get_mask(w2)
            x = self.activation_fn(F.linear(x, w1*m1, self.fc1.bias))
            x = F.linear(x, w2*m2, self.fc2.bias)
        else:
            x = self.activation_fn(self.fc1(x)*0.5)
            x = self.fc2(x)*0.5
        return x

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size, args=args
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, index=0
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if args.ff_block == 'ff':
            self.fc_block = FF_Block(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
            if args.num_stages > 1:
                if (index+1) % (math.ceil(args.decoder_layers/(args.num_stages))) == 0 and index+1 < args.decoder_layers:
                    self.bottleneck = True
                else:
                    self.bottleneck = False
        elif args.ff_block == 'bottleneck':
            self.fc_block = FF_Block(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
            #self.fc_block = FFBottleneck(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
            #self.fc_block = Linear8bitBlock(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
            #self.fc_block = FFBottleneck(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
            if args.num_stages > 1:
                if (index+1) % (math.ceil(args.decoder_layers/(args.num_stages))) == 0 and index+1 < args.decoder_layers:
                    self.bottleneck = FFBottleneck(args, self.embed_dim, args.decoder_ffn_embed_dim, self.activation_fn, float(activation_dropout_p))
                else:
                    self.bottleneck = None
        else:
            raise NotImplementedError(f'FF Block type not implemented: {args.ff_block}')

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        self.args = args
        self.index = index

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            args=args
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.fc_block(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if self.args.ff_block == 'bottleneck':
            if self.bottleneck is not None:
                x = self.bottleneck(x)
        elif self.args.ff_block == 'ff' and self.args.comm8bit:
            if self.bottleneck:
                x = Comm8bitFunction.apply(x)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
