# Normalization & Activation in Modern LLMs — Complete Notes

> Module notes: vanishing/exploding gradients in deep networks, Layer Normalization (from scratch),
> Pre-LN vs Post-LN gradient stability, RMSNorm derivation, gating mechanisms, and SwiGLU.

---

## Table of Contents

1. [Why Deep Networks Are Hard to Train](#1-why-deep-networks-are-hard-to-train)
2. [Vanishing and Exploding Gradients — From First Principles](#2-vanishing-and-exploding-gradients--from-first-principles)
3. [Layer Normalization (LayerNorm) — From Scratch](#3-layer-normalization-layernorm--from-scratch)
4. [Post-LN: The Original Transformer Placement](#4-post-ln-the-original-transformer-placement)
5. [Pre-LN: Moving Normalization Inside the Residual](#5-pre-ln-moving-normalization-inside-the-residual)
6. [Pre-LN vs Post-LN: Gradient Stability — Deep Analysis](#6-pre-ln-vs-post-ln-gradient-stability--deep-analysis)
7. [Ultra-Deep Networks: Where the Gap Between Pre-LN and Post-LN Becomes Critical](#7-ultra-deep-networks-where-the-gap-becomes-critical)
8. [RMSNorm: Simplifying LayerNorm](#8-rmsnorm-simplifying-layernorm)
9. [Activation Functions: Background Before SwiGLU](#9-activation-functions-background-before-swiglu)
10. [The Gating Mechanism](#10-the-gating-mechanism)
11. [SwiGLU: The Modern LLM Activation](#11-swiglu-the-modern-llm-activation)
12. [SwiGLU in the FFN Sub-Layer](#12-swiglu-in-the-ffn-sub-layer)
13. [Comparison Tables](#13-comparison-tables)
14. [Key Takeaways](#14-key-takeaways)
15. [References](#15-references)

---

## 1. Why Deep Networks Are Hard to Train

### 1.1 The promise of depth

Neural networks gain **representational power** from depth. A shallow network (few layers) can in principle approximate any function (Universal Approximation Theorem), but it may need an *exponentially* wide layer to do so. Depth is a way to gain representational expressivity *efficiently* — each layer builds abstractions on top of the previous one, and the number of parameters needed to represent the same function grows only polynomially with depth instead of exponentially.

This is why virtually every major LLM is very deep: GPT-2 uses 48 layers, LLaMA-2-70B uses 80 layers, Gemini Ultra reportedly uses over 100 transformer blocks. Greater depth → better language modeling, up to the limits imposed by compute budgets.

### 1.2 The fundamental training challenge: signals must flow across every layer

Training a neural network by gradient descent requires computing how the loss $\mathcal{L}$ (a scalar measuring how wrong the model's predictions are) changes with respect to every single weight $w$ in the network — i.e., computing $\partial \mathcal{L} / \partial w$ for all $w$ simultaneously, via backpropagation. Backpropagation is just repeated application of the **chain rule**: gradients at layer $l$ are computed from gradients at layer $l+1$ (the layer above it, closer to the loss), multiplied by the local Jacobian of layer $l$.

If you have $L$ layers stacked:

```
x_0 → Layer 1 → x_1 → Layer 2 → x_2 → ... → Layer L → Loss
```

then the gradient for Layer 1's parameters involves a product of Jacobians across all $L-1$ layers between Layer 1 and the Loss:

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdot \frac{\partial x_{L-1}}{\partial x_{L-2}} \cdots \frac{\partial x_2}{\partial x_1} \cdot \frac{\partial x_1}{\partial W_1}$$

This is a product of $(L-1)$ matrices (the per-layer Jacobians). Products of many matrices are numerically unstable by nature — they tend to either:
- collapse toward zero as $L$ grows ("vanishing gradients"), or
- blow up toward infinity ("exploding gradients").

Both outcomes make training fail. This is the **deep network training problem**, and it gets strictly worse as $L$ increases. At the extreme depth of modern LLMs (80, 100, 120 layers), it is the central obstacle that every architectural choice in this module is designed to address.

---

## 2. Vanishing and Exploding Gradients — From First Principles

### 2.1 A toy derivation: purely linear network

Start with the simplest case: a purely linear network with $L$ layers, no activations, just matrix multiplies:

$$x_l = W_l x_{l-1}, \qquad l = 1, \ldots, L$$

The backward pass gives:

$$\frac{\partial \mathcal{L}}{\partial x_0} = W_L^\top W_{L-1}^\top \cdots W_1^\top \frac{\partial \mathcal{L}}{\partial x_L}$$

This is a product of $L$ matrices. Consider the eigenvalues of each $W_l$ (or, for non-square matrices, the singular values). If the average singular value of $W_l$ is:

- **Less than 1** (say, $\sigma < 1$): the product's "effective scale" $\sim \sigma^L \to 0$ exponentially. With $L=100$ layers and $\sigma=0.95$: $0.95^{100} \approx 0.0059$. The gradient reaching layer 1 is less than 1% of what left the loss. This is **vanishing gradient**.
- **Greater than 1** (say, $\sigma > 1$): the product's scale $\sim \sigma^L \to \infty$ exponentially. With $L=100$ and $\sigma=1.05$: $1.05^{100} \approx 131.5$. The gradient explodes across layers. This is **exploding gradient**.
- **Exactly 1**: gradients propagate without shrinking or growing. This is called **isometric** propagation, and it's what normalization layers are trying to approximate.

The "safe" zone (singular values exactly 1) is a razor-thin target. Small deviations from it compound exponentially across layers.

### 2.2 The nonlinear case: saturation makes things worse

Add a nonlinearity (activation function) $\sigma(\cdot)$ after each linear layer: $x_l = \sigma(W_l x_{l-1})$. Now the backward pass through layer $l$ multiplies by the **Jacobian of $\sigma$ with respect to its input**, which is a diagonal matrix of element-wise derivatives:

$$\frac{\partial \sigma(z)}{\partial z} = \text{diag}\left(\sigma'(z_1), \sigma'(z_2), \ldots, \sigma'(z_d)\right)$$

For classic activations like sigmoid ($\sigma(z) = 1/(1+e^{-z})$) or tanh: the derivative $\sigma'(z)$ is bounded between 0 and 0.25 (for sigmoid) or 0 and 1 (for tanh), but crucially it **approaches 0 when $|z|$ is large** — the "saturating" regime. This is why early deep networks with sigmoid activations were almost impossible to train: on top of the weight-product shrinkage, you're also multiplying by numbers close to 0 at each layer wherever neurons saturated.

ReLU ($\max(0, z)$, derivative exactly 0 or 1) partially fixed this by having a non-saturating positive regime, but introduced new problems (dying ReLU, where units get permanently stuck at 0). The story of activation functions is partly a story of trying to get gradient magnitudes as close to 1 as possible across the entire forward pass.

### 2.3 Residual connections: the key structural fix

The transformative insight from ResNets (He et al., 2015): **add a direct "skip" connection** that passes the input around (bypasses) the main computation at each block:

$$x_{l+1} = x_l + F(x_l)$$

where $F$ is the block's function (attention + FFN for a transformer). Now the gradient of the loss with respect to $x_l$ is:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \left(I + \frac{\partial F(x_l)}{\partial x_l}\right)$$

The crucial term is $\frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot I$ — the **identity matrix directly passes the gradient through**. No matter how bad $\partial F / \partial x_l$ is (vanishing or exploding), the gradient has an "express route" that doesn't go through $F$ at all. This means that even if $F$'s contribution to the gradient is near-zero (vanishing), at least the identity term passes a copy of the gradient unchanged. Residual connections alone dramatically reduce (but do not eliminate) vanishing gradient problems.

**Why "but do not eliminate":** the residual connection bypasses *one* block, but you still need to propagate from the last block back to the first through $L$ multiplications of $(I + \partial F / \partial x_l)$ terms. If those Jacobians' largest eigenvalues are $> 1$, gradients can still explode. And over $L$ layers, even $(1 + \epsilon)^L$ blows up if $L$ is large enough. This is where **normalization** enters.

---

## 3. Layer Normalization (LayerNorm) — From Scratch

### 3.1 Why normalize at all?

The core problem residual connections don't fully solve: the *distribution* of activations $x_l$ can drift as the network gets deeper. If layer $l$ outputs activations with a large variance, the weight matrices in layer $l+1$ see very large-magnitude inputs, which (a) saturates activations, causing gradient vanishing, and (b) makes the network's behavior highly sensitive to weight initialization. We need to **control the distribution of activations** flowing through the network at each layer.

Batch Normalization (Ioffe & Szegedy, 2015) solved this for CNNs by normalizing *across the batch dimension* — for each feature, subtract its batch mean and divide by its batch standard deviation. This works beautifully when you have large, stable batches. For transformers and autoregressive language modeling, it does not work:
- Batch sizes in LLM training are often very small per GPU (1–4 sequences), making batch statistics noisy.
- At inference time with batch size 1, batch statistics are meaningless (you'd be normalizing each feature by itself).
- Sequence lengths vary, making the "spatial" dimensions inconsistent.

**Layer Normalization** (Ba et al., 2016) is the fix: instead of normalizing across the batch, normalize **within each single token's feature vector**, independently. Each token normalizes itself.

### 3.2 LayerNorm: formal definition

Given a vector $x \in \mathbb{R}^d$ (one token's activation vector at some layer), LayerNorm computes:

**Step 1: compute the mean and variance *across the feature dimension*:**

$$\mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2$$

**Step 2: normalize:**

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

where $\varepsilon \approx 10^{-6}$ is a tiny constant preventing division by zero.

**Step 3: affine rescale with learned parameters $\gamma$ (scale) and $\beta$ (shift):**

$$\text{LayerNorm}(x)_j = \gamma_j \hat{x}_j + \beta_j$$

$\gamma, \beta \in \mathbb{R}^d$ are vectors of learnable parameters (one scale and one shift per dimension), initialized to $\gamma = \mathbf{1}$, $\beta = \mathbf{0}$ (identity at init — does nothing at first, then the network learns to use them). The reason for this affine step is that raw normalization to zero mean and unit variance might be too constraining — the network might need to represent a distribution that genuinely has a non-zero mean or non-unit variance to best serve the downstream computation. $\gamma$ and $\beta$ restore this freedom.

**Critical properties:**
- Operates on one token at a time (no cross-batch, no cross-token dependencies).
- After normalization (before the affine step), every token's feature vector has exactly mean 0 and variance 1.
- Works identically during training and inference regardless of batch size.
- Adds $2d$ learnable parameters per LayerNorm instance.

### 3.3 What LayerNorm does for gradient flow

From the backward-pass perspective: LayerNorm constrains the *magnitude* of activations passing through it. Since post-normalization activations have variance close to 1 (the $\gamma$ can scale this, but $\gamma$ is initialized to 1 and doesn't change catastrophically in early training), the Jacobian of LayerNorm with respect to its input has eigenvalues that are also controlled in magnitude. This prevents the runaway accumulation of scale that causes exploding gradients.

More precisely: the chain rule through a LayerNorm layer produces a gradient that is **projected onto the subspace orthogonal to the mean direction** (because LayerNorm subtracts the mean, which kills the component of the gradient pointing in the all-ones direction) and **divided by the standard deviation** (which rescales the gradient magnitude to be roughly unit-scale). This makes the effective Jacobians of each layer approximately "unit-scale" regardless of what the activations were doing before normalization.

---

## 4. Post-LN: The Original Transformer Placement

### 4.1 How it works

In the original "Attention Is All You Need" (Vaswani et al., 2017), Layer Normalization is placed **after** the residual addition. The full expression for one sub-layer (say, multi-head attention) is:

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Attention}(x_l))$$

Or in two sub-layers (attention then FFN), the complete transformer block in Post-LN:

```
x → MultiHeadAttention → (+x) → LayerNorm → SubLayer 1 output
                                                    ↓
                         → FFN →  (+previous) → LayerNorm → Block output
```

Schematically: compute the sub-layer function on the input, add the residual (skip connection), *then* normalize.

### 4.2 Why Post-LN was used in the original paper

It was a natural and principled application of normalization: "put it where the signal enters the next operation, to ensure normalized inputs to each sub-layer." BERT, GPT-2, and many early transformer models used Post-LN.

### 4.3 The fundamental problem with Post-LN at large depth

Here is the key issue, and it requires thinking carefully about what the residual stream looks like during training, specifically at **initialization**.

At initialization, the weights of the attention and FFN sub-layers are small random matrices. The function $F(x_l) = \text{Attention}(x_l)$ or $F(x_l) = \text{FFN}(x_l)$ therefore outputs a vector whose magnitude is **small relative to the input** $x_l$ (since weights are initialized to be near-zero). So immediately after initialization:

$$\text{output of residual addition} = x_l + F(x_l) \approx x_l + \text{(small noise)}$$

The residual sum is *dominated by* $x_l$. Now apply LayerNorm to this sum. LayerNorm computes the mean and variance of $x_l + F(x_l)$. Since $F(x_l)$ is small noise relative to $x_l$, the normalization is essentially normalizing $x_l$ — so the LayerNorm output is approximately $\hat{x}_l$ (normalized version of $x_l$), with the contribution of $F$ almost entirely washed out. The network isn't learning anything meaningful from the sub-layer at init — the sub-layer's output is a small perturbation that gets absorbed into the renormalization.

**The deeper problem: gradient magnitudes are not controlled at initialization.** In Post-LN, the gradient flowing into $x_l$ through the residual path is:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial \text{LN output}} \cdot \frac{\partial \text{LN}(x_l + F(x_l))}{\partial x_l}$$

The derivative of LayerNorm with respect to its input includes a $1/\sigma$ factor (where $\sigma$ is the standard deviation of the *input to LN*, which is $x_l + F(x_l)$). In deep networks, $x_l$ builds up in magnitude as it passes through many residual additions — each layer adds something to the running sum in the residual stream. The residual stream's magnitude grows with depth. Therefore, $\sigma$ (the std-dev of the LN input) also grows with depth. And $1/\sigma$ — the scale of the gradient — **shrinks with depth**.

This means that in Post-LN, the gradient of the loss with respect to the input of a deeper block is *smaller* than that with respect to a shallower block. In very deep networks (many tens of layers), this gradient shrinkage across depth is severe — the earliest layers receive nearly-zero gradients, and those layers' parameters barely move during training. This is the **Post-LN vanishing gradient problem**, and it grows worse the deeper the network is.

**Practical consequence:** very deep Post-LN transformers require careful learning-rate warm-up schedules (start from an extremely small learning rate and ramp up slowly), layer-specific learning rate scaling, and are generally brittle to train. Attempting to train a 100-layer Post-LN transformer with a normal learning rate typically causes training instability or divergence.

---

## 5. Pre-LN: Moving Normalization Inside the Residual

### 5.1 How it works

**Pre-LN** (also called "Pre-Norm") places LayerNorm **before** the sub-layer function, *inside* the residual branch but before the function $F$ is applied:

$$x_{l+1} = x_l + F\!\left(\text{LayerNorm}(x_l)\right)$$

Or for the two-sub-layer transformer block:

```
x → LayerNorm → MultiHeadAttention → (+x) → SubLayer 1 output
                                                    ↓
                → LayerNorm → FFN  → (+previous) → Block output
```

The entire sub-layer function $F$ operates on a normalized version of the input. The residual connection adds directly back to the *un-normalized* input $x_l$, bypassing the LayerNorm.

### 5.2 What changes structurally

In Post-LN: LayerNorm sees $x_l + F(x_l)$ (the sum of clean input and sub-layer output).  
In Pre-LN: LayerNorm sees only $x_l$ (the clean residual-stream value, before the sub-layer contributes anything).

This single positional change has a profound effect on gradient flow, which we derive next.

### 5.3 Pre-LN at initialization

At initialization, $F(\text{LayerNorm}(x_l))$ is again a small vector (weights are near-zero). But now note what happens to the residual stream:

$$x_{l+1} = x_l + F(\text{LayerNorm}(x_l)) \approx x_l \quad \text{(at init)}$$

At initialization, every layer in Pre-LN is approximately an **identity function** — the residual output is almost exactly the input. This means the entire network (all $L$ layers stacked) is close to the identity map at initialization: $x_L \approx x_0$. This is called the **identity initialization property** or "lazy initialization regime."

Why is this good? It means the network starts from a near-identity state (essentially "doing nothing" and just passing the input through), which is a much more benign starting point for optimization than the chaotic behavior Post-LN networks exhibit at initialization. Early gradients correspond to small perturbations around the identity, and the network can begin learning gently from this stable starting point.

---

## 6. Pre-LN vs Post-LN: Gradient Stability — Deep Analysis

### 6.1 The gradient path in Pre-LN

Compute the gradient of loss with respect to the residual stream at layer $l$, $\partial \mathcal{L} / \partial x_l$. Starting from layer $L$ and applying the chain rule across the $L - l$ layers between $l$ and $L$:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k}$$

In Pre-LN, $x_{k+1} = x_k + F_k(\text{LN}(x_k))$, so:

$$\frac{\partial x_{k+1}}{\partial x_k} = I + \frac{\partial F_k(\text{LN}(x_k))}{\partial x_k} = I + J_k^F \cdot J_k^{LN}$$

where $J_k^F$ is the Jacobian of the sub-layer $F_k$ with respect to its normalized input, and $J_k^{LN}$ is the Jacobian of LayerNorm with respect to $x_k$.

Now, the product of these per-layer Jacobians across $L - l$ layers:

$$\prod_{k=l}^{L-1}\left(I + J_k^F J_k^{LN}\right)$$

**The key:** there is always an $I$ term in each factor. No matter how small or large $J_k^F J_k^{LN}$ is, the product always has the identity as a direct path through. Expanding this product, the lowest-order term is simply $I^{L-l} = I$ — meaning a copy of the loss gradient $\partial \mathcal{L} / \partial x_L$ arrives at $x_l$ essentially unchanged (apart from higher-order perturbations). Gradients flow through the residual stream's "highway" — the identity connections — without any forced multiplicative shrinkage tied to the depth $L$.

**This is the core stability guarantee of Pre-LN:** even as $L \to \infty$, there is always a gradient path from the loss to every layer that goes through only identity matrices. The gradient at layer 1 is at least as large as the gradient at layer $L$, up to the sub-layer correction terms.

### 6.2 Why Post-LN doesn't have this property

In Post-LN, $x_{k+1} = \text{LN}(x_k + F_k(x_k))$, so:

$$\frac{\partial x_{k+1}}{\partial x_k} = J_k^{LN}\left(I + J_k^F\right)$$

Notice: now the entire Jacobian (identity term included) is **multiplied by $J_k^{LN}$**. There is no "pure identity path" through $J_k^{LN}$ — the LayerNorm gates the residual skip connection itself. And as noted in Section 4.3, $J_k^{LN}$ shrinks in magnitude as depth increases (because the residual stream $x_k$ grows in magnitude with depth, and LayerNorm's derivative scales as $\sim 1/\|x_k\|$). So in Post-LN, the identity term — the supposedly "safe gradient highway" — gets shrunk by every LayerNorm in the chain. The more layers you have, the more shrinkage accumulates.

### 6.3 The residual stream magnitude: a key difference

In Pre-LN, the residual stream $x_l$ grows slightly with each layer (each $F$ adds something to it), but LayerNorm never directly touches the residual stream itself — it only normalizes a copy of it before feeding into $F$. So the residual stream can grow gradually, but its growth doesn't feed back into the gradient magnitude in a destabilizing way.

In Post-LN, the residual stream is normalized *back to unit variance* after every sub-layer. You might think this prevents the residual stream from growing — and you'd be right — but this forced normalization means that the effective "signal" in the residual stream is constantly being reset, making it harder for the network to accumulate useful information across many layers. It also means that early layers' contributions get normalized away as subsequent layers add to and then re-normalize the stream.

### 6.4 Why Post-LN can actually achieve better *final quality* than Pre-LN (the tradeoff)

Despite its gradient instability, Post-LN has an advantage: because LayerNorm is applied to the *output* of each sub-layer before it enters the next, the outputs are consistently normalized, which can give each layer a consistently clean, well-conditioned input. Models trained with Post-LN and careful warm-up schedules sometimes achieve better final quality than Pre-LN — the normalization at each sub-layer output effectively makes each layer's *learned function* more expressive and consistent. Several research results found that BERT-style (Post-LN) models perform slightly better than Pre-LN counterparts in the mid-depth regime (12-24 layers) when given careful training.

**The practical conclusion:** for very deep networks (50+ layers), **Pre-LN is essentially mandatory** for stable training without heroic engineering. For moderate depth (12-24 layers), Post-LN can work with care. Modern LLMs (LLaMA, Mistral, GPT-NeoX, GPT-4, etc.) all use Pre-LN — depth wins, and stability is non-negotiable at scale.

### 6.5 Summary of the tradeoff in one sentence

**Post-LN**: excellent per-layer output quality and expressiveness, but gradient highway is gated by shrinking LayerNorm Jacobians as depth grows → training instability in very deep networks.  
**Pre-LN**: identity gradient highway is clean and unobstructed by LayerNorm, enabling stable training at arbitrary depth, at the cost of slightly less constrained per-layer outputs.

---

## 7. Ultra-Deep Networks: Where the Gap Becomes Critical

### 7.1 The empirical failure mode of Post-LN at depth

The consequences of Post-LN's gradient instability become acute at depth in several ways:

**Gradient norm disparity**: measure $\|\partial \mathcal{L} / \partial x_l\|$ at each layer $l$ during early training. In Post-LN, this norm decreases roughly geometrically with how close $l$ is to the input: layer 96 of a 96-layer model might have 10,000× larger gradients than layer 1. The earliest layers barely move. Effectively, a deep Post-LN network often ends up being trained only by its *top* layers, with the bottom layers frozen in near-initialization states.

**Learning rate sensitivity**: to avoid exploding the top layers' gradients (which are large), you must use a tiny learning rate. But then the bottom layers' gradients (already tiny) effectively become zero. You can't win with a uniform learning rate.

**The warm-up necessity**: Post-LN requires learning rate warm-up specifically to cope with this — at initialization, the gradients for the top layers are unstable (they're large and depend sensitively on the un-normalized residual stream), so you need to start from an almost-zero learning rate and increase it slowly, giving the early layers a chance to develop some gradient signal before the top layers start running away. Pre-LN doesn't need warm-up nearly as aggressively, because gradients are naturally controlled from the start (near-identity initialization from Section 5.3).

### 7.2 How Pre-LN concretely mitigates vanishing gradients

At initialization, the residual stream in Pre-LN is approximately a direct pass-through of the input token embeddings: $x_L \approx x_0$. As a result, the gradient $\partial \mathcal{L} / \partial x_0 \approx \partial \mathcal{L} / \partial x_L$, meaning the embedding layer gets essentially the same gradient signal as the final layer. Every layer in between is learning simultaneously and at roughly comparable rates from the very first training step.

As training progresses and the sub-layer functions develop non-trivial outputs, the $J^F J^{LN}$ correction terms grow but remain regulated: the LayerNorm inside the residual branch (which acts on a clean copy of $x_l$) ensures that $J^{LN}$ doesn't shrink catastrophically with depth (because $x_l$ in the Pre-LN residual stream has a more stable magnitude than in the Post-LN case). The product of correction terms can still compound, but they compound around an identity baseline, not multiplicatively shrinking the baseline itself.

### 7.3 Residual stream perspective in transformer notation

It is useful to think of the transformer's **residual stream** as a single vector that accumulates contributions from every layer. Each attention sub-layer and each FFN sub-layer *reads from* and *writes to* this stream:

```
residual_stream = embedding(tokens)          # initialization
for each layer l:
    residual_stream += Attention(LN(residual_stream))     # Pre-LN attention write
    residual_stream += FFN(LN(residual_stream))           # Pre-LN FFN write
output = LN(residual_stream)                 # final normalization before unembedding
```

In this view, the residual stream is never modified *in-place* by LayerNorm — it only ever grows via additive writes from the sub-layers. LayerNorm is only applied to *views* of the stream (reads), not to the stream itself. The final LayerNorm at the output is what ensures a well-conditioned input to the unembedding (language model head). This "residual stream as an accumulator" perspective makes it visually clear why Pre-LN preserves gradient paths: every sub-layer's write goes to the same stream, and the loss gradient can flow backward through the stream's additions without any multiplicative gating by LayerNorm.

---

## 8. RMSNorm: Simplifying LayerNorm

### 8.1 Motivation: what is LayerNorm actually doing?

LayerNorm does two things in sequence:
1. **Re-centering**: subtract the mean $\mu$ from each element, ensuring the mean of the output is 0.
2. **Re-scaling**: divide by the standard deviation $\sigma$, ensuring the variance of the output is 1.

Then it optionally *un-does* some of this with learned $\gamma$ and $\beta$. The question asked by RMSNorm (Root Mean Square Layer Normalization, Zhang & Sennrich, 2019) is: **do we actually need step 1 (the mean subtraction)?**

### 8.2 The hypothesis: re-centering is not doing useful work

The argument from Zhang & Sennrich: the key stabilizing effect of LayerNorm is the **re-scaling** (making magnitudes comparable), not the **re-centering** (removing the mean). The mean subtraction enforces that the output has exactly zero mean, but in practice, with the subsequent $\gamma/\beta$ learned rescaling and the model having strong representational capacity, the network can represent any mean-shifted distribution regardless — it doesn't need LayerNorm to enforce zero-mean for correctness.

Furthermore, from a gradient-flow perspective: the main role of normalization is to control the *scale* of activations (which directly controls the scale of gradients, via the chain rule). Dividing by the RMS (root mean square) accomplishes this. Subtracting the mean first makes the RMS calculation slightly more accurate (you're computing std-dev relative to the mean, not relative to zero), but the practical difference is small, especially when the activations tend to have small means relative to their standard deviation (which is common in transformer embeddings).

### 8.3 RMSNorm: formal definition

Given $x \in \mathbb{R}^d$, RMSNorm skips the mean subtraction entirely. It computes the **Root Mean Square** of the raw vector:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2}$$

Then normalizes and rescales:

$$\text{RMSNorm}(x)_j = \frac{x_j}{\text{RMS}(x) + \varepsilon} \cdot \gamma_j$$

where $\varepsilon$ is a small stability constant and $\gamma \in \mathbb{R}^d$ is the learned scale vector (same role as $\gamma$ in LayerNorm). Note: there is **no $\beta$ (shift) parameter** in RMSNorm — re-centering is entirely removed, and the shift parameter would be redundant anyway if we're not enforcing zero-mean output.

Comparing side by side:

| | **LayerNorm** | **RMSNorm** |
|---|---|---|
| Compute mean? | Yes: $\mu = \frac{1}{d}\sum x_j$ | **No** |
| Normalize by | $\sqrt{\text{Var}(x) + \varepsilon} = \sqrt{\frac{1}{d}\sum(x_j-\mu)^2 + \varepsilon}$ | $\text{RMS}(x) + \varepsilon = \sqrt{\frac{1}{d}\sum x_j^2 + \varepsilon}$ |
| Learnable params | $\gamma \in \mathbb{R}^d$, $\beta \in \mathbb{R}^d$ | $\gamma \in \mathbb{R}^d$ only |
| Invariance | Shift-invariant (output same if you add a constant to $x$) + scale-invariant | Scale-invariant only |
| Parameters | $2d$ | $d$ |

### 8.4 Why RMSNorm is faster in practice

The speed advantage comes from two sources:

**1. Fewer arithmetic operations:** LayerNorm requires two passes over the $d$-dimensional vector: one to compute the mean, one to compute variance (which requires knowing the mean). RMSNorm requires only *one pass*: sum of squares, one square root, then elementwise divide. On modern hardware, this is a meaningful reduction in memory bandwidth pressure (you touch the $d$-dimensional vector fewer times), not just in floating-point operations.

**2. Simpler graph for fused kernels:** GPU kernels for normalization are often implemented as "fused" operations (the normalization and rescaling are done in a single kernel pass to avoid redundant global memory reads/writes). A simpler computation (one pass, no mean) means the fused kernel is less complex and easier for the compiler to optimize. Flash Attention and similar fused attention kernels have analogous motivations — fewer passes over memory is the key.

Reported speedups from RMSNorm vs LayerNorm in practice: roughly 10–40% faster normalization operations, which translates to a few percent speedup in overall training/inference throughput.

### 8.5 Gradient properties of RMSNorm vs LayerNorm

The gradient of RMSNorm with respect to its input $x$ is:

$$\frac{\partial \text{RMSNorm}(x)_j}{\partial x_k} = \frac{\gamma_j}{\text{RMS}(x)}\left(\delta_{jk} - \frac{x_j x_k}{d \cdot \text{RMS}(x)^2}\right)$$

(where $\delta_{jk}$ is the Kronecker delta — 1 if $j=k$, 0 otherwise). This is a rank-1 update to a scaled identity matrix, structurally similar to LayerNorm's Jacobian (which has an analogous form but with an additional mean-subtraction term). Both Jacobians have the property of projecting out one or two specific directions from the gradient, but RMSNorm only projects out the "all-equal" (scale) direction, while LayerNorm also projects out the "mean" direction.

For gradient stability purposes, both behave essentially identically in the Pre-LN setting: they both control the scale of activations reaching the sub-layer functions, and they both produce well-conditioned Jacobians that don't shrink or inflate gradients pathologically. RMSNorm's slight asymmetry (not mean-centering) has not been found to hurt training stability in practice.

### 8.6 Where RMSNorm is used

RMSNorm replaced LayerNorm in virtually every major LLM starting around 2022-2023: LLaMA, LLaMA-2, LLaMA-3, Mistral, Qwen, Falcon, DeepSeek, Gemma. The combination of Pre-LN + RMSNorm has become the dominant normalization strategy in open LLMs, trading a negligible amount of re-centering-induced invariance for a nontrivial throughput improvement at scale.

---

## 9. Activation Functions: Background Before SwiGLU

Before understanding SwiGLU, we need to understand what problem it solves. Let's trace the evolution of activation functions used in the FFN (feed-forward network) sub-layer of transformers.

### 9.1 The FFN sub-layer: structure

Every transformer block contains a **Feed-Forward Network (FFN)** sub-layer, applied independently to each token's representation. In the original Transformer:

$$\text{FFN}(x) = W_2 \cdot \sigma\!\left(W_1 x + b_1\right) + b_2$$

where $W_1 \in \mathbb{R}^{d_{ff} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{ff}}$, $d$ is the model dimension, and $d_{ff}$ is the FFN intermediate size (typically $4d$). $\sigma$ is an activation function applied elementwise.

The FFN is essentially a two-layer MLP expanded into a wider hidden dimension and then projected back. It's the "memory" component of each transformer block — attention mixes information between tokens, FFN processes each token's representation independently.

### 9.2 ReLU: the baseline

$$\text{ReLU}(x) = \max(0, x)$$

Properties:
- Derivative: 0 if $x < 0$, 1 if $x > 0$. Never saturates for positive $x$ → no gradient vanishing in the positive regime.
- Computationally extremely cheap: just a clamp.
- **Dying ReLU problem**: if a neuron's pre-activation is always negative (e.g., due to weight initialization), its gradient is always 0 — that neuron permanently stops learning. In deep networks with many neurons, a significant fraction can "die" this way.
- The "kink" at 0: ReLU is not differentiable at 0, and the hard threshold produces sparse, discontinuous activations. This is fine for optimization (subgradients work) but limits the function class the network can represent smoothly.

### 9.3 GeLU: a smooth, probabilistic approximation to ReLU

$$\text{GeLU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the standard normal CDF: $\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$.

Interpretation: GeLU outputs the input $x$ scaled by the *probability that a standard Gaussian exceeds $x$*. For large positive $x$, $\Phi(x) \to 1$ and $\text{GeLU}(x) \approx x$ (behaves like identity/ReLU). For large negative $x$, $\Phi(x) \to 0$ and $\text{GeLU}(x) \approx 0$ (like ReLU). For $x$ near 0, the transition is **smooth and differentiable** — unlike ReLU's hard kink.

GeLU is practically approximated as:

$$\text{GeLU}(x) \approx 0.5 x \left(1 + \tanh\!\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)$$

or with PyTorch's exact erf implementation: `F.gelu(x)`.

GeLU was adopted in BERT and GPT-2 and proved substantially better than ReLU for language models, particularly in terms of final quality metrics. The smooth gating behavior (where the activation "smoothly decides" to pass the input or suppress it, based on the input's value) seems to match the kind of computation useful in language modeling better than the hard threshold.

### 9.4 The key question GeLU asks: can we do better with an explicit gate?

GeLU essentially multiplies the input $x$ by a value-dependent weight $\Phi(x) \in [0, 1]$. But $\Phi(x)$ is determined by $x$ itself — the signal is gating *itself*. What if we separated the gating signal from the "value" signal? That is, what if one part of the computation decided *how much* to pass, and a separate part provided *what content* to pass? This is the motivating question for **Gated Linear Units (GLUs)** and, ultimately, **SwiGLU**.

---

## 10. The Gating Mechanism

### 10.1 What is a gate?

A **gate** is a mechanism that controls how much information flows through a pathway. It takes a control signal and outputs a multiplier (usually between 0 and 1) that scales some other signal:

$$\text{output} = \underbrace{v(x)}_{\text{content}} \cdot \underbrace{g(x)}_{\text{gate}}$$

where $v(x)$ is the "value" signal (what information to potentially pass) and $g(x)$ is the "gate" signal (how much of the value to actually let through). The two can be computed independently (from different projections of $x$), which is the key expressiveness advantage over a plain activation function like ReLU, where the same pre-activation determines both whether to activate and what value to output.

Gates have a long history in recurrent architectures (LSTM's forget/input/output gates, GRU's update/reset gates). Bringing them into feedforward computations for transformers is the innovation in GLU-family activations.

### 10.2 Sigmoid-Gated Linear Unit (GLU)

The original **GLU** (Dauphin et al., 2017) splits the FFN's linear projection into two halves and gates one with a sigmoid of the other:

$$\text{GLU}(x, W, V) = \sigma(xW) \odot xV$$

where $\sigma$ is the sigmoid function ($\sigma(z) = 1/(1+e^{-z})$, outputting values in $(0,1)$), $\odot$ is elementwise multiplication, $W, V \in \mathbb{R}^{d \times d_{ff}}$ are two separate learned projection matrices. The sigmoid of $xW$ acts as a data-dependent gate (values close to 0 suppress, values close to 1 pass), and $xV$ provides the content that gets selectively passed.

Compared to a plain FFN ($W_2 \sigma(W_1 x)$) with the same "hidden size" $d_{ff}$: GLU requires two projection matrices $W, V$ for the "first layer" instead of one, making it nominally more expensive — but empirically it achieves better quality *per parameter* that it actually uses, and you can often reduce the hidden size $d_{ff}$ slightly to keep total parameter count equivalent.

### 10.3 Why gating helps: the representational argument

In a plain FFN layer, each neuron in the hidden layer computes $\text{Act}(w_i \cdot x + b_i)$, and the activation function $\text{Act}$ applies the same fixed nonlinearity to the pre-activation. The decision of "whether this neuron should be active" and "what value it contributes" are both determined by the same inner product $w_i \cdot x$.

With a gated unit, you have:

- A "content path" $v(x) = V x$: a linear transformation that computes what values to potentially contribute.
- A "gate path" $g(x) = \sigma(W x)$: a separate linear transformation followed by a squashing function that computes *how much* of the content to pass.

Because $W$ and $V$ are different weight matrices, the gate can respond to entirely different features of $x$ than the content. The gate can say "yes, pass the content" based on contextual signals that are orthogonal to the content itself. This is more flexible than a plain activation, where the same feature value must serve as both the gate-trigger and the content-value simultaneously.

---

## 11. SwiGLU: The Modern LLM Activation

**Reference:** Noam Shazeer, 2020. "GLU Variants Improve Transformers."

SwiGLU combines the gating architecture of GLU with the **Swish** activation function in the gate path (instead of sigmoid), giving a smoother, better-performing variant.

### 11.1 The Swish activation function

$$\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

with $\beta = 1$ as the default (sometimes written SiLU — Sigmoid Linear Unit — when $\beta=1$):

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Properties of Swish/SiLU:
- **Non-monotonic**: unlike ReLU and GeLU, Swish is *not* strictly increasing everywhere. For $x < 0$, Swish dips slightly negative before rising toward 0. This negative region (roughly $x \in (-5, 0)$) is small but important: it means the activation can output small negative values for moderately negative inputs, rather than hard-zeroing like ReLU. This allows small negative contributions to backpropagate, addressing the dying-neuron problem.
- **Smooth everywhere**: no kinks, fully differentiable.
- **Self-gated**: $\text{Swish}(x) = x \cdot g(x)$ where $g(x) = \sigma(x)$ is determined by $x$ itself — the value is gating itself (like GeLU). This is what SwiGLU extends into a *two-component* (separate value and gate) formulation.

### 11.2 SwiGLU: the definition

SwiGLU replaces the sigmoid gate in GLU with the Swish activation (specifically SiLU with $\beta=1$):

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV) = \left(xW \cdot \sigma(xW)\right) \odot xV$$

Breaking this down:
- $xW$: first linear projection, the "gate logit" — this determines how active each unit should be.
- $\text{Swish}(xW)$: apply Swish to the gate logit. Now this signal is non-linear, smooth, and can output both small negative and positive values. It acts as the gate.
- $xV$: second separate linear projection, the "value" — the content to pass.
- $\odot$: elementwise multiply the gate and the value. The gate *selectively passes* the value.

The two paths ($W$ and $V$) are **learned separately** from the same input $x$, so the network can independently decide "what computation is relevant here" (via $V$, linear in $x$) and "how much of each unit should activate" (via $W$ and the Swish nonlinearity). Shazeer's paper empirically showed SwiGLU outperformed GLU (sigmoid gate), ReGLU (ReLU gate), and GEGLU (GeLU gate) variants on downstream NLP benchmarks.

### 11.3 The "SiGLU" / "SiLU" naming confusion

You will often see this called **SiLU** gate in code, because `torch.nn.functional.silu` is the implementation used, and SiLU = Swish with $\beta=1$. The naming goes: GLU (sigmoid gate) → SwiGLU/SiGLU (SiLU/Swish gate) — all from the same GLU family, just with different gating activation functions.

---

## 12. SwiGLU in the FFN Sub-Layer

### 12.1 Full SwiGLU FFN formulation

Modern LLMs (LLaMA, Mistral, GPT-4-era models, Falcon, Qwen, DeepSeek, etc.) replace the original FFN:

$$\text{FFN}_{\text{original}}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$

with:

$$\text{FFN}_{\text{SwiGLU}}(x) = W_2 \left(\text{SiLU}(W_1 x) \odot (W_3 x)\right)$$

where:
- $W_1 \in \mathbb{R}^{d_{ff} \times d}$ — "gate" projection (feeds into SiLU)
- $W_3 \in \mathbb{R}^{d_{ff} \times d}$ — "value" projection (feeds directly as content)
- $W_2 \in \mathbb{R}^{d \times d_{ff}}$ — "down" projection (projects back to model dimension)

This requires **3 weight matrices** instead of the original 2, at the same intermediate size $d_{ff}$.

### 12.2 The parameter-count adjustment trick

Because SwiGLU uses 3 matrices instead of 2, and to maintain roughly the same total parameter count as an original FFN with $d_{ff} = 4d$, LLaMA and similar models set their FFN hidden dimension to:

$$d_{ff} = \frac{2}{3} \times 4d = \frac{8d}{3}$$

(approximately; implementations often round this to a multiple of 64 or 256 for GPU efficiency). So you have 3 matrices of size $d \times (8d/3)$ rather than 2 matrices of size $d \times 4d$: $3 \times (8d/3)d = 8d^2$ vs $2 \times 4d \times d = 8d^2$. Equivalent parameter count.

For example, LLaMA-2's actual intermediate size: for $d=4096$, the official value is $d_{ff}=11008$ (vs the "naive" $4 \times 4096 = 16384$ you'd use without the 2/3 adjustment).

### 12.3 Why SwiGLU outperforms GeLU

Several complementary explanations have been proposed:

**1. Separate content and gate paths:** As discussed in Section 10.3, the explicit separation of "what value to compute" ($W_3$) and "whether to use it" ($W_1$ + SiLU) gives the network more representational flexibility than a plain activation function where both roles are played by the same pre-activation. The network can learn to compute a rich set of intermediate values and then have a completely separate learned controller decide which of those values are actually useful for the current input.

**2. Smooth gating with non-trivial negative regime:** SiLU's small negative region allows gating values to be softly "almost-off" without being hard-zero. This provides a richer gradient signal in the near-off regime compared to ReLU's hard zero or sigmoid's near-zero saturation. Training signals can still flow through near-zero gated units, preventing the dead-neuron problem.

**3. Multiplicative interactions:** The $\odot$ operation between $W_1 x$ and $W_3 x$ creates multiplicative interactions between features. Two features can jointly determine the output (if feature A is high AND feature B is high, the output is high) — a type of interaction a pure MLP with additive activations approximates only implicitly. Multiplicative interactions have long been known to be efficient representations of certain language regularities (compositionality, feature conjunction).

**4. Empirical evidence:** Shazeer (2020) tested six GLU variants (sigmoid, tanh, ReLU, GELU, SiLU, bilinear gates) on T5-style pretraining and found SwiGLU/SiLU-gate consistently achieved best perplexity and downstream task performance. PaLM, LLaMA, and subsequent models adopted it and reported similar findings — SwiGLU-FFN is now effectively the standard.

### 12.4 PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Three weight matrices instead of two
        # d_ff is already adjusted (typically ~8/3 * d_model, not 4 * d_model)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # gate path
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # value path
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # down-projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: SiLU applied to first projection
        gate = F.silu(self.W1(x))         # shape: (..., d_ff)
        # Value: linear of second projection
        value = self.W3(x)                 # shape: (..., d_ff)
        # Elementwise product of gate and value
        hidden = gate * value              # shape: (..., d_ff)
        # Down-project back to model dimension
        return self.W2(hidden)             # shape: (..., d_model)

# The complete Pre-LN transformer block with RMSNorm + SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x):
        # Pre-LN: normalize BEFORE the sub-layer, add to residual AFTER
        x = x + self.attn(self.norm1(x))   # attention with Pre-LN
        x = x + self.ffn(self.norm2(x))    # FFN with Pre-LN
        return x

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))   # learned scale, no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS across feature dimension
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma
```

### 12.5 Bias-free FFN layers

Modern LLMs (LLaMA, Mistral) also drop the bias terms from FFN projections entirely (`bias=False` in the example above). The rationale: with RMSNorm and the learned $\gamma$ scale, bias terms in the FFN are redundant and add parameters without improving quality. Removing them also slightly simplifies the computation graph and the interaction between normalization and linear layers.

---

## 13. Comparison Tables

### 13.1 Pre-LN vs Post-LN

| Criterion | Post-LN (original Transformer) | Pre-LN (modern LLMs) |
|---|---|---|
| **Placement** | LN after residual addition: `LN(x + F(x))` | LN before sub-layer: `x + F(LN(x))` |
| **Gradient highway** | Gated by $J^{LN}$, which shrinks with depth | Clean identity path, $J^{LN}$ does not gate the skip connection |
| **Gradient vanishing at depth** | Severe (grows worse as $L$ increases) | Mild (identity residual always provides usable gradient) |
| **At initialization** | Sub-layer output nearly washed out by LN; chaotic gradients at top layers | Network ≈ identity function; all layers start with similar gradient magnitudes |
| **LR warm-up requirement** | Necessary; without it, training often diverges | Much weaker warm-up needed, often none at scale |
| **Final quality (moderate depth)** | Slightly higher (more constrained intermediate outputs) | Slightly lower at equivalent depth, but often overtaken by going deeper |
| **Final quality (very deep, >50L)** | Often fails to train without careful engineering | Trains reliably; dominant choice |
| **Used by** | BERT, GPT-2, original Transformer, T5 (variant) | LLaMA, GPT-NeoX, PaLM, Mistral, Qwen, Falcon, almost all post-2022 LLMs |
| **Additional final LN?** | Not typically needed | Yes: a final `LN(residual_stream)` before the output head is standard, since Pre-LN's residual stream is not normalized at the final layer |

### 13.2 LayerNorm vs RMSNorm

| Criterion | LayerNorm | RMSNorm |
|---|---|---|
| **Computes mean?** | Yes | No |
| **Normalization denominator** | $\sqrt{\text{Var}(x) + \varepsilon}$ | $\sqrt{\text{Mean}(x^2) + \varepsilon}$ |
| **Learnable parameters** | $\gamma, \beta$ ($2d$) | $\gamma$ only ($d$) |
| **Invariances** | Scale-invariant + shift-invariant | Scale-invariant only |
| **Speed** | Baseline | ~10–40% faster (one fewer pass; simpler fused kernel) |
| **Gradient stability** | Comparable in Pre-LN setting | Comparable in Pre-LN setting |
| **Practical quality difference** | Baseline | Empirically comparable; widely used at scale without quality loss |
| **Used by** | BERT, GPT-2, T5, early transformers | LLaMA, LLaMA-2/3, Mistral, Falcon, Qwen, Gemma, DeepSeek |

### 13.3 Activation function comparison in FFN

| | **ReLU** | **GeLU** | **SwiGLU** |
|---|---|---|---|
| **Formula** | $\max(0, x)$ | $x\Phi(x)$ | $\text{SiLU}(W_1 x) \odot W_3 x$ |
| **Smooth?** | No (kink at 0) | Yes | Yes |
| **Negative outputs?** | No | Tiny (near $x=0$) | Small but intentional |
| **Gate type** | Self-gate (hard) | Self-gate (probabilistic) | Separate learned gate + value |
| **Number of projection matrices** | 2 ($W_1, W_2$) | 2 ($W_1, W_2$) | 3 ($W_1, W_2, W_3$) |
| **Parameter parity trick** | — | — | Reduce $d_{ff}$ to $\approx 8d/3$ |
| **Dying neuron risk** | High | Low | Very low |
| **Multiplicative interactions** | No | No | Yes ($W_1 \odot W_3$ paths) |
| **Quality (LLM scale)** | Lowest | Good | Best (empirically) |
| **Used by** | Original Transformer | BERT, GPT-2, GPT-3 | LLaMA, PaLM, Mistral, DeepSeek, Qwen |

---

## 14. Key Takeaways

- Deep networks require gradients to flow backward through many multiplicative chain-rule factors. Products of many matrices naturally either vanish (→ 0) or explode (→ ∞) unless each factor is carefully controlled to be approximately unit-scale. Residual connections provide an identity-matrix bypass, but don't solve everything at large depth.
- **Post-LN** places LayerNorm after the residual addition, which normalizes the *skip connection* itself. In deep networks, the growing residual stream magnitude causes LayerNorm's Jacobian to shrink, gating the identity bypass and causing severe gradient vanishing at early layers. Post-LN works for moderate depths but fails at scale without heroic engineering.
- **Pre-LN** places LayerNorm inside the residual branch (before the sub-layer function), leaving the skip connection itself unaffected by LayerNorm. This preserves a direct identity gradient path from the loss back to every layer simultaneously. At initialization, a Pre-LN network is approximately the identity function, enabling stable, concurrent learning across all layers from the very first step.
- **RMSNorm** drops LayerNorm's mean-subtraction step, normalizing by the root mean square instead of the standard deviation. This removes the $\beta$ shift parameter, halves the learned parameters per norm, and enables significantly faster fused GPU kernels. Gradient stability is equivalent to LayerNorm in the Pre-LN setting. It is now the dominant normalization in open LLMs.
- **SwiGLU** replaces the single two-matrix FFN ($W_2\,\text{Act}(W_1 x)$) with a three-matrix gated formulation ($W_2(\text{SiLU}(W_1 x) \odot W_3 x)$), where $W_1$ (gate path) and $W_3$ (value path) are independent learned projections and their elementwise product creates multiplicative interactions. This separation of "what to pass" from "how much to pass" gives the network greater representational flexibility per parameter, empirically outperforming ReLU and GeLU activations consistently across model sizes.
- The canonical modern LLM block is **Pre-LN + RMSNorm + SwiGLU**: these three choices together make large-scale deep training stable, fast, and high quality. Almost every post-2022 open LLM (LLaMA, Mistral, DeepSeek, Qwen, Falcon, Gemma) uses exactly this combination.

---

## 15. References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* (Original Transformer; Post-LN.)
- Ba, J.L. et al. (2016). *Layer Normalization.* (LayerNorm.)
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* (RMSNorm.)
- He, K. et al. (2015). *Deep Residual Learning for Image Recognition.* (Residual connections.)
- Dauphin, Y. et al. (2017). *Language Modeling with Gated Convolutional Networks.* (GLU.)
- Shazeer, N. (2020). *GLU Variants Improve Transformers.* (SwiGLU, GEGLU, ReGLU family.)
- Elfwing, S. et al. (2018). *Sigmoid-Weighted Linear Units for Neural Network Function Approximation.* (SiLU/Swish original paper.)
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* (GeLU.)
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture.* (Formal analysis of Pre-LN vs Post-LN gradient stability.)
- Touvron, H. et al. (2023). *LLaMA 2: Open Foundation and Fine-Tuned Chat Models.* (Pre-LN + RMSNorm + SwiGLU in practice.)