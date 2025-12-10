# Experimental Investigation of Computational Non-Determinism in Deep Learning

## Abstract

This document presents a systematic experimental investigation into the sources, magnitude, and mitigation of computational non-determinism in deep learning systems. Three research questions (RQs) are conduncted targeting: (1) the quantification and origin of inference non-determinism in Transformer-based LLMs arising from parallel batch processing, (2) the performance trade-offs of enforcing strict deterministic execution, and (3) the efficacy of stochastic rounding as an alternative numerical mitigation strategy. All experiments are conducted on a single NVIDIA RTX 3070 (8 GB VRAM) using PyTorch. My findings demonstrate that standard inference exhibits significant logit divergence due to non-associative floating-point accumulation in attention layers, that determinism incurs prohibitive latency overhead (~25x vs eager execution), and that stochastic rounding offers a practical path to stable low-precision training without strict bitwise reproducibility.

---

## 1. Introduction

### 1.1 Background and Motivation

The reproducibility crisis in computational science has underscored the importance of deterministic computation for scientific rigor. In deep learning, non-determinism manifests as bitwise differences in model outputs across identical runs, arising from the fundamental interaction between:

1. **Floating-Point Non-Associativity**: IEEE 754 arithmetic is non-associative; $(a + b) + c \neq a + (b + c)$ due to finite precision rounding.
2. **Parallel Execution Scheduling**: GPU thread scheduling is non-deterministic, causing variable accumulation orders in reduction operations (e.g., summation in matrix multiplication, softmax normalization).

This non-determinism is particularly acute in Transformer architectures, where attention mechanisms involve intensive parallel reductions over sequence and feature dimensions. The problem is further exacerbated by *dynamic batching* in inference servers, where the same input processed in different batch contexts yields numerically distinct outputs—violating the principle of *batch invariance*.

### 1.2 Research Questions

This investigation addresses three research questions:

| RQ | Focus | Key Metric |
|----|-------|------------|
| **RQ1** | Magnitude and origin of inference non-determinism | Max logit difference (solo vs. batched) |
| **RQ2** | Performance cost of deterministic execution | Latency overhead (%) |
| **RQ3** | Efficacy of stochastic rounding for training stability | Final validation accuracy |

### 1.3 Experimental Environment

All experiments are conducted on the following hardware and software stack:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 3070 (8 GB GDDR6) |
| CUDA | 12.x |
| PyTorch | ≥ 2.1 |
| Transformers | ≥ 4.39 |
| Triton | ≥ 2.2 |
| Precision | bfloat16 (RQ1/RQ2), fp32 (RQ3) |

---

## 2. RQ1: The Magnitude and Origin of Inference Non-Determinism

### 2.1 Research Question

> How does parallel batch processing introduce computational non-determinism in Transformer-based LLMs, and is it possible to achieve bitwise identity between batched and solo inference?

### 2.2 Hypothesis

Standard PyTorch eager execution uses optimized CUDA kernels (cuBLAS, cuDNN) that employ parallel reductions with non-deterministic accumulation order. When processing the same input in different batch contexts (solo vs. batched with distractors), the attention layer's softmax and matrix multiplication operations will produce numerically distinct outputs due to:

1. Different memory layouts causing different reduction orders
2. Padding-induced misalignment in batched sequences
3. Adaptive kernel selection based on tensor shapes

We hypothesize that by enforcing fixed-order accumulation via custom Triton kernels, bitwise identity can be achieved.

### 2.3 Methodology

#### 2.3.1 Model and Inputs

- **Model**: GPT-2 Large (774M parameters) loaded with `attn_implementation="eager"` to ensure standard PyTorch attention without Flash Attention.
- **Precision**: bfloat16 (16-bit brain floating-point)
- **Prompts**: 10 diverse natural language prompts covering various domains (see Table 1)

**Table 1: Evaluation Prompts**

| ID | Prompt (excerpt) |
|----|------------------|
| 1 | "The quick brown fox jumps over..." |
| 2 | "Artificial intelligence is tra..." |
| 3 | "To be or not to be, that is th..." |
| 4 | "The grand unifying theory of p..." |
| 5 | "Python is a popular programmin..." |
| 6 | "In the beginning God created t..." |
| 7 | "A journey of a thousand miles ..." |
| 8 | "Quantum mechanics describes na..." |
| 9 | "The stock market experienced a..." |
| 10 | "Climate change poses a signifi..." |

#### 2.3.2 Experimental Protocol

For each prompt $p_i$, we compute:

1. **Solo inference**: Process $p_i$ alone (batch size = 1)
2. **Batched inference**: Process $p_i$ alongside a long distractor sentence (batch size = 2)

The distractor is a 5x-repeated sentence designed to force significant padding on the target prompt, inducing memory layout misalignment.

We extract the logit vector at the last real token position (ignoring padding) and compute:

$$\text{Diff} = \max_j |L_{\text{solo}}[j] - L_{\text{batched}}[j]|$$

This is performed under two execution modes:

1. **Standard Mode**: Default PyTorch eager execution
2. **Batch Invariant Mode**: Custom Triton kernels with fixed-order accumulation

#### 2.3.3 Batch Invariant Implementation

The batch invariant implementation (`experiments/BatchInvariant.py`) replaces the following PyTorch operations with custom Triton kernels:

| Operation | Standard Kernel | Batch Invariant Kernel |
|-----------|-----------------|------------------------|
| `mm` / `addmm` | cuBLAS GEMM | Persistent matmul with fixed tile order |
| `bmm` | cuBLAS batched GEMM | Sequential per-batch matmul |
| `softmax` | cuDNN fused kernel | Two-pass log-softmax with sequential reduction |
| `mean` | CUDA reduction | Block-wise sequential accumulation |

The key innovation is enforcing a deterministic tile traversal order in matrix multiplication and sequential (rather than tree-based parallel) reduction in softmax/mean operations.

### 2.4 Results

**Table 2: RQ1 Results — Logit Divergence (Max Absolute Difference)**

| Run | Prompt Excerpt | Diff (Standard) | Diff (Invariant) | Success |
|-----|----------------|-----------------|------------------|---------|
| 1 | "The quick brown fox..." | 0.250 | 0.0 | ✓ |
| 2 | "Artificial intelligence..." | 0.125 | 0.0 | ✓ |
| 3 | "To be or not to be..." | 0.125 | 0.0 | ✓ |
| 4 | "The grand unifying..." | 0.125 | 0.0 | ✓ |
| 5 | "Python is a popular..." | 0.125 | 0.0 | ✓ |
| 6 | "In the beginning God..." | 0.250 | 0.0 | ✓ |
| 7 | "A journey of a thousand..." | 0.125 | 0.0 | ✓ |
| 8 | "Quantum mechanics..." | 0.250 | 0.0 | ✓ |
| 9 | "The stock market..." | 0.125 | 0.0 | ✓ |
| 10 | "Climate change poses..." | 0.250 | 0.0 | ✓ |
| **Average** | — | **0.175** | **0.0** | **100%** |

### 2.5 Analysis

The results unequivocally demonstrate:

1. **Standard execution exhibits significant non-determinism**: The average max absolute logit difference of 0.175 (in bfloat16 scale) represents a substantial numerical divergence. Given that bfloat16 has ~3 significant decimal digits, a difference of 0.125–0.250 can meaningfully alter softmax probabilities and thus token predictions.

2. **Non-determinism is batch-context dependent**: The same model weights and input tokens produce different outputs solely based on co-batched inputs. This violates the intuitive expectation that $f(x)$ should equal $f(x)$ regardless of what other inputs are processed alongside $x$.

3. **Bitwise identity is achievable**: The batch invariant implementation achieves exactly zero difference across all 10 test cases, confirming that enforcing fixed-order accumulation eliminates batch-induced non-determinism.

4. **Root cause is attention layer reductions**: The primary source of divergence is the attention mechanism's softmax (reducing over key dimension) and value aggregation (reducing over sequence dimension). These operations involve $O(n^2)$ or $O(nd)$ parallel reductions where $n$ is sequence length and $d$ is embedding dimension.

---

## 3. RQ2: The Performance Trade-off of Deterministic Execution

### 3.1 Research Question

> What is the latency overhead associated with enforcing strict deterministic execution compared to standard optimized kernels?

### 3.2 Methodology

#### 3.2.1 Baseline Configurations

We benchmark three execution modes on GPT-2 Large:

| Mode | Attention Implementation | Determinism |
|------|--------------------------|-------------|
| **Eager** | Standard PyTorch (`attn_implementation="eager"`) | Non-deterministic |
| **SDPA** | Scaled Dot-Product Attention / Flash Attention (`attn_implementation="sdpa"`) | Non-deterministic |
| **Batch Invariant** | Custom Triton kernels (from RQ1) | Deterministic |

#### 3.2.2 Benchmark Protocol

- **Model**: GPT-2 Large (bfloat16)
- **Batch size**: 8 (simulating realistic inference server load)
- **Prompts**: Same 10 prompts as RQ1
- **Iterations**: 20 forward passes per configuration (after 3 warmup iterations)
- **Timing**: CUDA events for precise GPU timing

### 3.3 Results

**Table 3: RQ2 Results — Latency Comparison (ms per forward pass)**

| Run | Eager (ms) | Invariant (ms) | SDPA (ms) | Overhead vs Eager | Overhead vs SDPA |
|-----|------------|----------------|-----------|-------------------|------------------|
| 1 | 35.4 | 1214.1 | 27.1 | +3332% | +4375% |
| 2 | 34.2 | 860.5 | 27.5 | +2416% | +3025% |
| 3 | 34.1 | 841.7 | 24.4 | +2368% | +3350% |
| 4 | 34.0 | 847.4 | 24.2 | +2390% | +3401% |
| 5 | 34.1 | 929.1 | 24.2 | +2623% | +3736% |
| 6 | 34.2 | 845.4 | 26.6 | +2370% | +3076% |
| 7 | 34.5 | 845.1 | 27.5 | +2348% | +2978% |
| 8 | 34.6 | 1016.3 | 24.7 | +2834% | +4017% |
| 9 | 34.3 | 845.3 | 27.7 | +2363% | +2957% |
| 10 | 34.5 | 845.6 | 24.5 | +2353% | +3346% |
| **Average** | **34.4** | **909.1** | **25.8** | **+2540%** | **+3426%** |

### 3.4 Analysis

1. **Determinism incurs prohibitive overhead**: The batch invariant implementation is approximately **26x slower** than eager execution and **35x slower** than SDPA. This overhead renders strict determinism impractical for production inference at scale.

2. **SDPA provides significant speedup**: Flash Attention (SDPA) is ~33% faster than eager attention, demonstrating the value of fused kernels. However, this optimization is fundamentally incompatible with determinism due to its parallel reduction strategy.

3. **Trade-off is fundamental**: The performance gap is not an implementation artifact but reflects a fundamental tension between:
   - **Parallelism**: Modern GPUs achieve throughput via massive parallelism with non-deterministic scheduling
   - **Determinism**: Fixed-order accumulation requires serialization, negating parallelism benefits

4. **Overhead sources**:
   - Sequential matmul for `bmm` (no batched parallelism)
   - Two-pass softmax (separate max-finding and normalization passes)
   - No kernel fusion (each operation launches separately)

### 3.5 Complementary Study: CNN Convolution Determinism

To provide context beyond Transformers, we also benchmark deterministic convolution in a CNN setting.

**Script**: `experiments/RQ2_Complement.py`

**Setup**: Single Conv2d layer (64 channels, 3×3 kernel, 32×32 input) benchmarked in "fast" (cuDNN benchmark enabled) vs "deterministic" (`torch.use_deterministic_algorithms(True)`) modes.

**Table 4: Convolution Determinism Overhead**

| Batch Size | Fast (ms) | Deterministic (ms) | Overhead | Throughput Loss |
|------------|-----------|--------------------|---------:|----------------:|
| 1 | 0.055 | 0.077 | +39% | -28% |
| 16 | 0.141 | 0.124 | -12%* | +15%* |
| 64 | 0.442 | 0.549 | +24% | -20% |

*Note: At batch size 16, deterministic mode happens to select a faster algorithm for this specific shape.

**Interpretation**: CNN convolution determinism carries a more moderate overhead (1.1–1.4x) compared to Transformer attention (25x+). This is because:
- Convolution involves fewer reduction dimensions
- cuDNN provides dedicated deterministic algorithms for common conv shapes
- The parallel reduction in conv is less extensive than in attention's sequence-length dimension

---

## 4. RQ3: Efficacy of Stochastic Rounding for Training Stability

### 4.1 Research Question

> To what extent can stochastic rounding mitigate output variance and improve training stability in low-precision settings, and how does this compare to enforcing bitwise determinism?

### 4.2 Motivation

Given that strict determinism is prohibitively expensive (RQ2), we investigate an alternative approach: **stochastic rounding**. Rather than demanding bitwise reproducibility, stochastic rounding provides *statistical* stability by ensuring that rounding errors are unbiased in expectation.

In standard round-to-nearest:
$$\text{round}(x) = \lfloor x + 0.5 \rfloor$$

This introduces systematic bias toward representable values, causing gradient information loss in low-precision training.

In stochastic rounding:
$$\text{SR}(x) = \begin{cases} \lfloor x \rfloor & \text{with probability } \lceil x \rceil - x \\ \lceil x \rceil & \text{with probability } x - \lfloor x \rfloor \end{cases}$$

The expected value $\mathbb{E}[\text{SR}(x)] = x$, preserving gradient information in expectation.

### 4.3 Methodology

#### 4.3.1 Model and Dataset

- **Model**: ResNet-18 (classifier head reset to 10 classes)
- **Dataset**: CIFAR-10 (50,000 train / 10,000 test images, 32×32 pixels)
- **Training**: SGD with momentum (lr=0.1, momentum=0.9, weight decay=5e-4, Nesterov), 10 epochs, batch size 128

#### 4.3.2 Quantization Scheme

We simulate 16-bit fixed-point arithmetic by quantizing activations, weights, and biases in each convolution layer:

$$q(x) = \frac{\text{round}(x \cdot 2^b)}{2^b}$$

where $b = 8$ fractional bits, giving a quantization step of $2^{-8} \approx 0.004$.

#### 4.3.3 Experimental Conditions

| Mode | Rounding Method | Precision |
|------|-----------------|-----------|
| **fp32** | None (full precision baseline) | 32-bit float |
| **nearest** | Round-to-nearest | Simulated 16-bit fixed-point |
| **stochastic** | Stochastic rounding | Simulated 16-bit fixed-point |

All conditions start from identical random initialization (controlled seeds).

### 4.4 Results

**Table 5: RQ3 Results — Training Performance**

| Mode | Best Val Acc | Final Val Acc | Training Time (s) |
|------|-------------:|---------------|------------------:|
| fp32 | 71.03% | 69.71% | 203.1 |
| nearest | 71.26% | 71.26% | 209.5 |
| stochastic | **72.06%** | **72.06%** | 232.1 |

**Figure 1: Validation Accuracy Trajectories**

```
Epoch:     1     2     3     4     5     6     7     8     9    10
─────────────────────────────────────────────────────────────────
fp32:    41.2  53.4  58.8  65.2  66.6  65.3  69.8  68.9  71.0  69.7
nearest: 36.6  45.9  50.6  57.0  59.9  65.7  68.6  68.5  66.6  71.3
stoch:   37.5  50.6  59.1  64.3  64.4  66.7  64.8  70.6  69.5  72.1
```

### 4.5 Analysis

1. **Stochastic rounding achieves highest accuracy**: Despite the quantization constraint, stochastic rounding outperforms both the fp32 baseline (+1.0 percentage point) and nearest rounding (+0.8 percentage point) after 10 epochs.

2. **Convergence dynamics differ**:
   - **fp32**: Fastest early convergence but shows overfitting symptoms (best at epoch 9, drops at epoch 10)
   - **nearest**: Slower start due to biased gradient truncation, but stabilizes
   - **stochastic**: Slower initial progress (unbiased noise slows convergence) but achieves superior final performance

3. **Stochastic rounding preserves gradient signal**: The unbiased nature of stochastic rounding ensures that small gradient updates are not systematically lost to quantization, enabling continued learning even in low-precision regimes.

4. **Practical implications**: Stochastic rounding offers a middle ground between:
   - Full precision (expensive memory, deterministic)
   - Nearest rounding (cheap, biased, potentially unstable)
   - Strict determinism (prohibitively slow per RQ2)

5. **Training time overhead**: Stochastic rounding adds ~14% overhead compared to fp32 due to random number generation for each rounding decision. This is far more acceptable than the 2500%+ overhead of deterministic kernels.

---

## 5. Discussion

### 5.1 Summary of Findings

| RQ | Key Finding | Implication |
|----|-------------|-------------|
| RQ1 | Standard inference exhibits 0.175 avg max logit diff between solo and batched processing; batch invariant kernels achieve exact 0.0 diff | Batch-induced non-determinism is real and measurable; it can be eliminated with custom kernels |
| RQ2 | Deterministic execution incurs ~2500% overhead vs eager, ~3400% vs SDPA | Strict determinism is impractical for production inference |
| RQ3 | Stochastic rounding achieves 72.1% accuracy vs 71.0% (fp32) and 71.3% (nearest) | Stochastic rounding provides a practical alternative to strict determinism for training |

### 5.2 Theoretical Interpretation

The experimental results align with the theoretical framework outlined in the project plan:

1. **Non-Associativity Manifests at Scale**: The logit differences observed in RQ1 (0.125–0.250) arise from accumulated rounding errors across the ~24 attention layers and ~774M parameters of GPT-2 Large. Each attention head performs $O(n^2 \cdot d)$ multiply-accumulate operations where reduction order varies with batch context.

2. **Hardware-Software Co-Design Trade-offs**: Modern GPU kernels (cuBLAS, cuDNN, Flash Attention) achieve performance through:
   - Parallel reduction with non-deterministic accumulation
   - Kernel fusion eliminating memory round-trips
   - Adaptive algorithm selection based on tensor shapes

   These optimizations are fundamentally at odds with determinism, which requires:
   - Fixed reduction order (sequential or carefully coordinated)
   - Separate kernel launches for each operation
   - Shape-independent algorithm choices

3. **Stochastic Rounding as Regularization**: The superior accuracy of stochastic rounding (RQ3) suggests it acts as implicit regularization, similar to dropout or gradient noise. The random rounding injects noise that prevents overfitting while preserving expected gradient values.

### 5.3 Practical Recommendations

Based on our findings, we recommend:

1. **For Debugging/Auditing**: Use batch invariant mode sparingly for debugging specific inference discrepancies. Accept the latency cost as a debugging tool, not a production configuration.

2. **For Production Inference**: Accept non-determinism as inherent. Design systems to be robust to small numerical variations (e.g., ensemble methods, temperature scaling, top-k sampling).

3. **For Training**: Consider stochastic rounding for low-precision training, especially on memory-constrained hardware. The accuracy benefits outweigh the modest overhead.

4. **For Reproducibility**: Fix all random seeds and use `torch.use_deterministic_algorithms(True)` for convolution-heavy models (acceptable overhead). For Transformer-heavy models, document expected variance ranges rather than demanding bit-exact reproduction.

### 5.4 Limitations and Future Work

1. **Model Scale**: Experiments use GPT-2 Large (774M params). Larger models (7B+) may exhibit different non-determinism patterns due to tensor parallelism and distributed execution.

2. **Hardware Specificity**: Results are specific to RTX 3070. Data center GPUs (A100, H100) with different memory hierarchies and SM counts may show different overhead ratios.

3. **Stochastic Rounding Scope**: RQ3 applies stochastic rounding only to convolution layers. Extending to attention layers and studying interaction with mixed-precision training (AMP) is future work.

4. **Dynamic Batching**: RQ1 uses synthetic batching. Real server workloads with dynamic request batching may exhibit different non-determinism patterns.

---

## 6. Conclusion

This investigation provides empirical evidence for the fundamental tension between computational efficiency and deterministic reproducibility in deep learning. We demonstrate that:

1. **Non-determinism is real and significant**: Standard Transformer inference produces measurably different outputs for identical inputs processed in different batch contexts.

2. **Determinism has a price**: Enforcing strict bitwise reproducibility incurs ~25x latency overhead, rendering it impractical for production deployment.

3. **Alternatives exist**: Stochastic rounding offers a practical path to stable low-precision training without demanding strict determinism, achieving superior accuracy through unbiased gradient preservation.

These findings inform the design of reproducible deep learning systems and highlight the need for nuanced approaches to reproducibility that balance scientific rigor with computational practicality.


## Appendix: Experiment Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `experiments/RQ1.py` | Quantify batch invariance divergence | `outputs/rq1_results.json` |
| `experiments/RQ2.py` | Benchmark deterministic overhead | `outputs/rq2_results.json` |
| `experiments/RQ2_Complement.py` | CNN convolution determinism benchmark | `outputs/rq2_complement_results.json` |
| `experiments/RQ3.py` | Stochastic rounding training study | `outputs/rq3_results.json` |
| `experiments/BatchInvariant.py` | Triton kernel implementations | (library) |

### Reproduction Commands

```bash
# RQ1: Batch invariance study
python experiments/RQ1.py

# RQ2: Deterministic overhead benchmark
python experiments/RQ2.py

# RQ2 Complement: CNN convolution benchmark
python experiments/RQ2_Complement.py

# RQ3: Stochastic rounding training
python experiments/RQ3.py --epochs 10 --batch-size 128
```