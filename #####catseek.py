"""
CATSEEKR2.0B – Simulated 2 Billion Parameter 4‑Bit LLM with DeepSeek‑R1 MLA
Optimizations:
  - LUT for instant INT4 unpacking.
  - Local variable caching.
  - List comprehensions for vector ops.
  - Multi‑layer support for realistic scaling.
  - Pure Python, no external files.
"""

import math
import random
import threading
import tkinter as tk
import builtins
import webbrowser

# =============================================================================
# 0. Optimized Pure Python Math Engine
# =============================================================================
# Pre‑computed LUT for INT4 unpacking (256 entries)
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

def silu(v):
    _exp = math.exp
    return [x / (1.0 + _exp(-x)) if abs(x) < 700 else 0.0 for x in v]

def softmax(v):
    if not v:
        return []
    _max, _exp = max, math.exp
    max_val = _max(v)
    exps = [_exp(x - max_val) for x in v]
    s = sum(exps)
    return [e / s for e in exps]

def rms_norm(v, weight, eps=1e-6):
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
    return [(x * inv_std) * w for x, w in zip(v, weight)]

# =============================================================================
# 1. REAL 4‑Bit Quantization (W4A8 Optimized)
# =============================================================================
def quantize_activation(v, bit_width=8):
    _max, _round, _min = max, round, min
    max_abs = _max(abs(x) for x in v) or 1e-5
    qmax = (1 << (bit_width - 1)) - 1
    scale = max_abs / qmax
    quantized = [int(_max(-qmax, _min(qmax, _round(x / scale)))) for x in v]
    return quantized, scale

def pack_weights_4bit(mat):
    _max, _round, _min = max, round, min
    max_val = _max(_max(abs(x) for x in row) for row in mat) or 1e-5
    scale = max_val / 7.0

    packed_mat = []
    for row in mat:
        packed_row = []
        for i in range(0, len(row), 2):
            w1, w2 = row[i], row[i+1] if i+1 < len(row) else 0.0
            q1 = int(_max(-8, _min(7, _round(w1 / scale))))
            q2 = int(_max(-8, _min(7, _round(w2 / scale))))
            packed = ((q1 & 0x0F) << 4) | (q2 & 0x0F)
            packed_row.append(packed)
        packed_mat.append(packed_row)
    return packed_mat, scale

class BitLinear:
    __slots__ = ['quant_mode', 'packed_weight', 'w_scale', 'weight']

    def __init__(self, in_f, out_f, quant_mode='4bit'):
        self.quant_mode = quant_mode
        float_weight = rand_matrix(out_f, in_f)

        if quant_mode == '4bit':
            self.packed_weight, self.w_scale = pack_weights_4bit(float_weight)
            self.weight = None
        else:
            self.weight = float_weight
            self.packed_weight = None

    def forward(self, x):
        if self.quant_mode == '4bit':
            x_q, x_scale = quantize_activation(x, bit_width=8)
            out_q = []
            lut = INT4_LUT
            w_scale = self.w_scale
            packed_w = self.packed_weight

            for packed_row in packed_w:
                acc = 0
                for i in range(len(packed_row)):
                    packed = packed_row[i]
                    q1, q2 = lut[packed]
                    idx = i << 1
                    acc += q1 * x_q[idx]
                    if idx + 1 < len(x_q):
                        acc += q2 * x_q[idx + 1]
                out_q.append(acc)

            combined_scale = w_scale * x_scale
            return vec_mul_scalar(out_q, combined_scale)
        else:
            return mat_vec_mul(self.weight, x)

# =============================================================================
# 2‑4. DeepSeek‑R1 Style Model (MLA + MoE) with multiple layers
# =============================================================================
class DeepSeekMLA:
    __slots__ = ['head_dim', 'w_down_kv', 'w_up_k', 'w_up_v', 'w_out']

    def __init__(self, dim, quant_mode):
        self.head_dim = dim // 4
        self.w_down_kv = BitLinear(dim, dim//4, quant_mode)
        self.w_up_k = BitLinear(dim//4, dim, quant_mode)
        self.w_up_v = BitLinear(dim//4, dim, quant_mode)
        self.w_out = BitLinear(dim, dim, quant_mode)

    def apply_rope(self, vec, pos):
        out = [0.0] * len(vec)
        _cos, _sin = math.cos, math.sin
        for i in range(0, len(vec)-1, 2):
            freq = 1.0 / (10000 ** (i / len(vec)))
            theta = pos * freq
            c, s = _cos(theta), _sin(theta)
            out[i] = vec[i] * c - vec[i+1] * s
            out[i+1] = vec[i+1] * c + vec[i] * s
        return out

    def forward(self, x, pos=0):
        c_kv = self.w_down_kv.forward(x)
        k = self.w_up_k.forward(c_kv)
        v = self.w_up_v.forward(c_kv)
        k = self.apply_rope(k, pos)

        score = sum(a * b for a, b in zip(x, k)) / math.sqrt(self.head_dim)
        attn_out = vec_mul_scalar(v, math.tanh(score))
        return self.w_out.forward(attn_out)

class Expert:
    __slots__ = ['up', 'down']
    def __init__(self, dim, quant_mode):
        self.up = BitLinear(dim, dim*2, quant_mode)
        self.down = BitLinear(dim*2, dim, quant_mode)

    def forward(self, x):
        up_out = self.up.forward(x)
        return self.down.forward([u * max(0, u) for u in up_out])

class CatR1XMoE:
    __slots__ = ['experts', 'router']
    def __init__(self, dim, quant_mode):
        self.experts = [Expert(dim, quant_mode) for _ in range(4)]
        self.router = BitLinear(dim, 4, quant_mode)

    def forward(self, x):
        weights = softmax(self.router.forward(x))
        out = [0.0] * len(x)
        for i, w in enumerate(weights):
            if w > 0.1:
                exp_out = self.experts[i].forward(x)
                out = vec_add(out, vec_mul_scalar(exp_out, w))
        return out

class DeepSeekBitNetModel:
    __slots__ = ['embed', 'mla_layers', 'moe_layers', 'head']

    def __init__(self, vocab_size=5000, dim=128, num_layers=12, quant_mode='4bit'):
        self.embed = rand_matrix(vocab_size, dim)
        self.mla_layers = [DeepSeekMLA(dim, quant_mode) for _ in range(num_layers)]
        self.moe_layers = [CatR1XMoE(dim, quant_mode) for _ in range(num_layers)]
        self.head = BitLinear(dim, vocab_size, quant_mode)

    def forward_token(self, token_id, pos):
        x = self.embed[max(0, min(token_id, len(self.embed)-1))]
        for mla, moe in zip(self.mla_layers, self.moe_layers):
            x = vec_add(x, mla.forward(x, pos))
            x = vec_add(x, moe.forward(x))
        return self.head.forward(x)

# =============================================================================
# 5. Codebase Cat R1 – Logic & Syntax Handler
# =============================================================================
class CodebaseCatR1:
    @staticmethod
    def get_ascii():
        return """     /\\_/\\     [ CATSEEKR2.0B – SIMULATED 2B PARAMETERS ]
    ( o.o )  > "Now with 12 layers and 128‑dim embeddings!"
     > ^ <"""

# =============================================================================
# 6. GUI (600x400) – Thread‑Safe with fixed print redirection
# =============================================================================
class CATSEEKR2_0BGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CATSEEKR2.0B – Optimized 4‑Bit LLM (Simulated 2B)")
        self.root.geometry("600x400")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        # Sidebar
        self.sidebar = tk.Frame(root, bg="#252526", width=150)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main = tk.Frame(root, bg="#1e1e1e")
        self.main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console = tk.Text(self.main, bg="#1e1e1e", fg="#d4d4d4",
                               font=("Consolas", 10), wrap=tk.WORD, state=tk.NORMAL)
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_box = tk.Entry(self.main, bg="#3c3c3c", fg="#ffffff",
                                  font=("Consolas", 11), insertbackground="white")
        self.input_box.pack(fill=tk.X, padx=5, pady=5)
        self.input_box.bind("<Return>", self.handle_input)

        # Settings
        self.lang_mode = tk.StringVar(value="auto")
        self.quant_mode = tk.StringVar(value="4bit")

        tk.Label(self.sidebar, text="OPTIONS", bg="#252526", fg="#cccccc").pack(pady=10)

        langs = [("Auto", "auto"), ("English", "en"), ("中文", "zh"), ("Random", "rand")]
        for t, v in langs:
            tk.Radiobutton(self.sidebar, text=t, variable=self.lang_mode, value=v,
                           bg="#252526", fg="#d4d4d4", selectcolor="#252526").pack(anchor="w", padx=10)

        tk.Label(self.sidebar, text="WEIGHTS", bg="#252526", fg="#cccccc").pack(pady=10)
        tk.Radiobutton(self.sidebar, text="Real 4‑Bit", variable=self.quant_mode, value="4bit",
                       bg="#252526", fg="#d4d4d4", selectcolor="#252526").pack(anchor="w", padx=10)

        btn = tk.Button(self.sidebar, text="Open DeepSeek Chat",
                        command=lambda: webbrowser.open("https://chat.deepseek.com"),
                        bg="#3c3c3c", fg="black", font=("Consolas", 10))
        btn.pack(pady=10, padx=10, fill=tk.X)

        # Replace builtins.print with a thread‑safe GUI print that handles end/sep/flush
        builtins.print = self.gui_print

        # Initial output
        print("CATSEEKR2.0B Optimized Kernel Loaded.")
        print(CodebaseCatR1.get_ascii())
        print("Simulated 2B parameters (dim=128, layers=12, vocab=5000).")
        print("Type 'print(1+1)' to test code interpreter.\n")

    def gui_print(self, *args, sep=' ', end='\n', flush=False):
        """Thread‑safe print replacement that outputs to the GUI console."""
        string = sep.join(str(a) for a in args) + end
        self.console.after(0, lambda: self.console.insert(tk.END, string))
        if flush:
            self.console.after(0, lambda: self.console.see(tk.END))

    def handle_input(self, event=None):
        text = self.input_box.get().strip()
        if not text:
            return
        self.input_box.delete(0, tk.END)
        print(f"\n>>> {text}")

        code_keywords = ("print(", "def ", "import ", "for ", "if ", "class ")
        if text.startswith(">>>") or any(k in text for k in code_keywords):
            threading.Thread(target=self.run_code_interpreter, args=(text,)).start()
        else:
            threading.Thread(target=self.run_chat_simulation, args=(text,)).start()

    def run_code_interpreter(self, code):
        print("\n<catr1_code_interpreter>")
        print("  <status>Compiling Python code...</status>")

        env = {'math': math, 'random': random}
        stdout_capture = []

        def custom_print(*args, **kwargs):
            stdout_capture.append(" ".join(map(str, args)))

        env['print'] = custom_print

        try:
            exec_code = code.replace(">>>", "").strip()
            exec(exec_code, env)

            if stdout_capture:
                print("  <output>")
                for line in stdout_capture:
                    print(f"    {line}")
                print("  </output>")
            else:
                print("  <output>None</output>")
        except Exception as e:
            print(f"  <error>{type(e).__name__}: {e}</error>")

        print("</catr1_code_interpreter>\n")

    def run_chat_simulation(self, prompt):
        lang = self.lang_mode.get()
        if lang == "rand":
            lang = random.choice(["en", "zh", "mix"])

        print("\n<catr1_reasoning>")
        print(f"  <detect_language mode='{lang}'/>")
        print(f"  <quantization bits='4' type='W4A8_LUT_OPTIMIZED'/>")
        print("  <internal_monologue>")
        print("    - Input received. Loading 2B parameter simulation.")
        print("    - Using 12 transformer layers with MLA + MoE.")
        print("    - LUT decompression active.")
        print("  </internal_monologue>")
        print("</catr1_reasoning>")

        print("\n<catr1_response>")

        # Use scaled‑down but plausible dimensions for a runnable simulation
        model = DeepSeekBitNetModel(vocab_size=5000, dim=128, num_layers=12,
                                    quant_mode=self.quant_mode.get())

        # Bilingual vocabulary (expanded a bit)
        vocab = ["meow", "喵", "speed", "速度", "optimized", "优化", "weight", "权重",
                 "scale", "规模", "cat", "猫", "python", "代码", "matrix", "矩阵",
                 "I", "think", "认为", "it", "is", "是", "fast", "快", "very", "非常",
                 "good", "好", "AI", "智能", "model", "模型", "2B", "二十亿"]

        token_id = sum(ord(c) for c in prompt) % 5000

        for i in range(12):  # Slightly longer response
            logits = model.forward_token(token_id, i)
            idx = logits.index(max(logits))
            word = vocab[idx % len(vocab)]
            token_id = idx
            print(word, end=" ", flush=True)

        print("\n</catr1_response>\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = CATSEEKR2_0BGUI(root)
    root.mainloop()
