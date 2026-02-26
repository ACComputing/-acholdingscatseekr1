"""
BitNetR1-2B – Pure Python 4-Bit BitNet-style 2B Parameter LLM (R1 Clone)
- Real INT4 quantization with LUT
- MLA + MoE (with proper causal attention)
- Console only (no tkinter, no files, pure Python)
"""

import math
import random
import threading
import builtins

# =============================================================================
# Optimized Pure Python Math Engine + INT4 LUT
# =============================================================================
INT4_LUT = []
for byte in range(256):
    q1 = (byte >> 4) & 0x0F
    q1 = q1 if q1 < 8 else q1 - 16
    q2 = byte & 0x0F
    q2 = q2 if q2 < 8 else q2 - 16
    INT4_LUT.append((q1, q2))

def rand_matrix(rows, cols, std=0.02):
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_mul_scalar(v, s):
    return [a * s for a in v]

def mat_vec_mul(mat, vec):
    return [sum(r * v for r, v in zip(row, vec)) for row in mat]

def softmax(v):
    if not v: return []
    max_val = max(v)
    exps = [math.exp(x - max_val) for x in v]
    s = sum(exps)
    return [e / s for e in exps]

def rms_norm(v, weight, eps=1e-6):
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
    return [(x * inv_std) * w for x, w in zip(v, weight)]

# =============================================================================
# 4-Bit Quantization (W4A8)
# =============================================================================
def quantize_activation(v, bit_width=8):
    max_abs = max(abs(x) for x in v) or 1e-5
    qmax = (1 << (bit_width - 1)) - 1
    scale = max_abs / qmax
    quantized = [max(-qmax, min(qmax, round(x / scale))) for x in v]
    return quantized, scale
#
