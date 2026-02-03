"""
Per-channel quantization utilities for MLPs using PyTorch's built-in functions.
"""

import torch
import torch.nn as nn
import copy


def quantize_linear_layer(layer: nn.Linear, n_bits: int = 8) -> nn.Linear:
    """
    Quantize a Linear layer's weights using per-output-channel quantization.
    Uses torch.fake_quantize_per_channel_affine under the hood.
    
    Parameters
    ----------
    layer : nn.Linear
        The linear layer to quantize.
    n_bits : int
        Number of bits for weight quantization.
    
    Returns
    -------
    nn.Linear
        A new linear layer with fake-quantized weights.
    """
    weight = layer.weight.data
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1
    
    # Compute per-channel scale (symmetric)
    amax = weight.abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / qmax
    zero_point = torch.zeros(weight.shape[0], dtype=torch.int64, device=weight.device)
    
    # Use PyTorch's built-in fake quantize
    dq_weight = torch.fake_quantize_per_channel_affine(
        weight, scale, zero_point, axis=0, quant_min=qmin, quant_max=qmax
    )
    
    # Create new layer with quantized weights
    new_layer = nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    new_layer.weight.data = dq_weight
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data.clone()
    
    return new_layer


def quantize_to(model: nn.Module, n: int = 8) -> nn.Module:
    """
    Quantize all Linear layers in a model using per-channel quantization.
    
    Parameters
    ----------
    model : nn.Module
        The model to quantize (e.g., an MLP from parity.py).
    n : int
        Number of bits for quantization (default: 8).
    
    Returns
    -------
    nn.Module
        A new model with quantized weights.
    
    Example
    -------
    >>> mlp = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    >>> mlp_q = quantize_to(mlp, n=4)  # 4-bit quantization
    
    Note
    ----
    For standard int8 dynamic quantization, you can also use PyTorch's built-in:
    
        import torch.ao.quantization as quant
        mlp_q = quant.quantize_dynamic(mlp, {nn.Linear}, dtype=torch.qint8)
    
    This function gives you control over bit-width (n) for experimentation.
    """
    model_q = copy.deepcopy(model)
    
    if isinstance(model_q, nn.Sequential):
        for i, layer in enumerate(model_q):
            if isinstance(layer, nn.Linear):
                model_q[i] = quantize_linear_layer(layer, n_bits=n)
    else:
        # Handle arbitrary nn.Module with named_modules
        for name, module in list(model_q.named_modules()):
            if isinstance(module, nn.Linear) and name:
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    parent = model_q
                    attr = parts[0]
                else:
                    parent = model_q.get_submodule(parts[0])
                    attr = parts[1]
                setattr(parent, attr, quantize_linear_layer(module, n_bits=n))
    
    return model_q


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Convenience wrapper for PyTorch's built-in dynamic int8 quantization.
    
    This is the simplest option if you just need standard int8 quantization.
    """
    import torch.ao.quantization as quant
    return quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def compute_quantization_error(
    model: nn.Module,
    model_q: nn.Module,
) -> dict[str, float]:
    """
    Compute the quantization error (MSE) for each Linear layer.
    
    Returns
    -------
    dict
        Mapping from layer index/name to MSE.
    """
    errors = {}
    
    if isinstance(model, nn.Sequential):
        for i, (layer, layer_q) in enumerate(zip(model, model_q)):
            if isinstance(layer, nn.Linear):
                mse = ((layer.weight - layer_q.weight) ** 2).mean().item()
                errors[f"layer_{i}"] = mse
    else:
        for (name, module), (_, module_q) in zip(
            model.named_modules(), model_q.named_modules()
        ):
            if isinstance(module, nn.Linear):
                mse = ((module.weight - module_q.weight) ** 2).mean().item()
                errors[name] = mse
    
    return errors
