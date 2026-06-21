# Rotary Position Embeddings (RoPE) — Complete Notes

> Module notes: positional encoding via rotation, relative-position emergence, long-term decay, and context-length extrapolation (Position Interpolation, NTK-aware scaling, YaRN, Su-scaled RoPE / LongRoPE).

---

## Table of Contents

1. [Why Transformers Need Position Information](#1-why-transformers-need-position-information)
2. [The Two Families of Position Encoding](#2-the-two-families-of-position-encoding)
3. [Complex Numbers Primer (From Scratch)](#3-complex-numbers-primer-from-scratch)
4. [RoPE in 2D: The Core Idea](#4-rope-in-2d-the-core-idea)
5. [Generalizing RoPE to d Dimensions](#5-generalizing-rope-to-d-dimensions)
6. [Why Relative Position Falls Out of the Dot Product](#6-why-relative-position-falls-out-of-the-dot-product)
7. [Practical Implementation (the "rotate-half" trick)](#7-practical-implementation-the-rotate-half-trick)
8. [The Long-Term Decay Property](#8-the-long-term-decay-property)
9. [The Extrapolation Problem](#9-the-extrapolation-problem)
10. [Position Interpolation (PI)](#10-position-interpolation-pi)
11. [NTK-Aware Scaling](#11-ntk-aware-scaling)
12. [YaRN (Yet another RoPE extensioN)](#12-yarn-yet-another-rope-extension)
13. [Su-Scaled RoPE / LongRoPE ("SuMA")](#13-su-scaled-rope--longrope-suma)
14. [Comparison Table](#14-comparison-table)
15. [Key Takeaways](#15-key-takeaways)
16. [References](#16-references)

---

## 1. Why Transformers Need Position Information

Self-attention computes a weighted sum over all tokens in a sequence, and the weighting (the softmax over $QK^T$) is a function of *content only* — it has no built-in notion of order. If you permute the input tokens, a pure self-attention layer (no positional signal) produces the same set of outputs, just permuted. This is the **permutation-equivariance** property of attention.

That's a problem because language is sequential: "dog bites man" and "man bites dog" must not be treated as the same bag of tokens. So every transformer needs *some* mechanism to inject position information into the computation.

There are two places you could inject it:

- **At the input embedding stage**: add or combine a position-dependent signal with the token embedding before it ever reaches attention. This is *absolute positional encoding* (APE).
- **Inside the attention computation itself**: modify how $Q$ and $K$ interact so that the attention score directly depends on the *distance* between two tokens. This is *relative positional encoding* (RPE).

RoPE is a relative encoding scheme, but with a twist: it is implemented as a transformation applied to $Q$ and $K$ individually (so it looks like it's adding absolute information), yet the math is engineered so that only the *relative* offset survives in the attention score. This hybrid nature is what makes RoPE elegant and is the central thing these notes will derive.

---

## 2. The Two Families of Position Encoding

### 2.1 Absolute Positional Encoding (APE)

**Sinusoidal (original Transformer, Vaswani et al. 2017):** each position $m$ gets a fixed vector built from sine/cosine waves of different frequencies:

$$PE(m, 2i) = \sin\left(\frac{m}{10000^{2i/d}}\right), \qquad PE(m, 2i+1) = \cos\left(\frac{m}{10000^{2i/d}}\right)$$

This vector is **added** to the token embedding before the first layer. It's parameter-free and deterministic, but the position information has to survive being added to content information, get pushed through every layer, and somehow still let the model infer *relative* distance — which the model has to learn indirectly.

**Learned APE (BERT, GPT-2, etc.):** a trainable embedding table indexed by position, added the same way. Works fine within the trained length but generalizes very poorly beyond it — the model never saw embeddings for position 5000 during training of a 2048-length model, so it has nothing meaningful to do with them.

**Shared weakness of all APE methods:** they encode position as an *absolute* index. The relationship between token at position 5 and token at position 10 has to be re-derived by the network from two absolute signals; it is not handed to the model directly.

### 2.2 Relative Positional Encoding (RPE)

**Shaw et al. (2018) / T5 relative bias / ALiBi:** these inject a *bias term* into the attention score that depends directly on $m - n$ (the distance between query position $m$ and key position $n$), instead of encoding absolute position into the vectors at all.

These work well but usually require either learned bias tables (extra parameters, awkward to extrapolate) or hand-designed penalty functions (ALiBi: a fixed linear penalty per head proportional to distance).

**RoPE's place in this picture:** RoPE achieves a relative encoding *effect* (attention score is a function of $m-n$ only) using a *multiplicative, parameter-free* transformation on $Q$ and $K$ — no extra bias table, no extra parameters, and (as we'll see) a built-in notion of decay with distance. This is why it became the dominant choice in LLaMA, GPT-NeoX, PaLM, Mistral, Qwen, and most modern open LLMs.

---

## 3. Complex Numbers Primer (From Scratch)

RoPE's cleanest derivation is in the language of complex numbers, so let's build that up from zero.

### 3.1 What is a complex number

A complex number is a pair of real numbers $(a, b)$ written as

$$z = a + bi, \qquad i = \sqrt{-1}$$

You can plot $z$ as a point on a 2D plane: $a$ on the horizontal (real) axis, $b$ on the vertical (imaginary) axis. So **every complex number is secretly just a 2D vector** $(a, b)$. This is the entire reason complex numbers are useful for RoPE: RoPE operates on *pairs* of dimensions in $Q$/$K$ vectors, and each pair can be treated as one complex number.

### 3.2 Polar form and Euler's formula

Any 2D point $(a, b)$ can also be written in polar coordinates: a radius $r$ (distance from origin) and an angle $\theta$ (from the positive real axis):

$$a = r\cos\theta, \qquad b = r\sin\theta \quad \Rightarrow \quad z = r(\cos\theta + i \sin\theta)$$

Euler's formula states:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

so $z = r e^{i\theta}$. This is just a more compact notation — same point, same vector, written using the exponential of an imaginary number instead of a sin/cos pair.

### 3.3 Multiplying complex numbers = rotating vectors

This is the single most important fact for RoPE. Take a complex number $z = r e^{i\theta}$ and multiply it by $e^{i\phi}$ (a "unit" complex number on the circle, i.e., radius 1):

$$z \cdot e^{i\phi} = r e^{i\theta} \cdot e^{i\phi} = r e^{i(\theta + \phi)}$$

The radius $r$ is unchanged. The angle simply adds $\phi$. **Multiplying by $e^{i\phi}$ rotates the vector by angle $\phi$, without changing its length.**

In matrix form, multiplying the 2D vector $(a, b)$ by $e^{i\phi}$ is identical to applying the 2D **rotation matrix**:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix}$$

You can verify this by expanding $(a+bi)(\cos\phi + i\sin\phi) = (a\cos\phi - b\sin\phi) + i(a\sin\phi + b\cos\phi)$, which matches the matrix output exactly.

**Key properties to remember going forward:**

- Rotation preserves vector length (so it preserves dot-product magnitudes in a controlled way — important later).
- Rotations **compose by adding angles**: rotating by $\phi_1$ then by $\phi_2$ is the same as rotating once by $\phi_1+\phi_2$.
- Rotating two vectors by the same angle preserves the angle *between* them. Rotating two vectors by *different* angles changes the angle between them by exactly the *difference* of the two rotation angles. This last fact is the seed of the entire RoPE relative-position trick.

---

## 4. RoPE in 2D: The Core Idea

Forget multi-dimensional embeddings for a moment. Suppose query and key vectors were just 2D: $q = (q_1, q_2)$ at position $m$, and $k = (k_1, k_2)$ at position $n$.

**RoPE's idea:** instead of adding a position signal, *rotate* $q$ by an angle proportional to its position $m$, and rotate $k$ by an angle proportional to its position $n$. Pick a fixed base frequency $\theta$ (a scalar, not to be confused with the rotation angle — frequency × position = angle):

$$q'_m = R(m\theta)\, q, \qquad k'_n = R(n\theta)\, k$$

where $R(\alpha)$ is the 2×2 rotation matrix from Section 3.3. Using complex notation, treating $q$ and $k$ as complex numbers $q = q_1 + iq_2$, $k=k_1+ik_2$:

$$q'_m = q \cdot e^{im\theta}, \qquad k'_n = k \cdot e^{in\theta}$$

Now compute the attention score as the real part of $q'_m \overline{k'_n}$ (complex conjugate dot product — this is the complex-number equivalent of the real dot product for 2D vectors):

$$q'_m \overline{k'_n} = \left(q \, e^{im\theta}\right)\overline{\left(k\, e^{in\theta}\right)} = q\bar{k}\, e^{im\theta} e^{-in\theta} = q\bar{k}\, e^{i(m-n)\theta}$$

Look closely: **the absolute positions $m$ and $n$ have vanished, and only the difference $(m-n)$ remains** inside the exponent. This is the entire magic trick of RoPE in one line. We rotated each vector by its own absolute position, but because rotations compose by *angle subtraction* when you take an inner product, only the *relative* rotation between the query and key survives.

---

## 5. Generalizing RoPE to d Dimensions

Real query/key vectors aren't 2D, they have hidden dimension $d$ per head (e.g., $d=64$ or $d=128$). RoPE handles this by **chopping the $d$-dimensional vector into $d/2$ pairs**, and treating each pair as an independent 2D rotation (independent complex number), each with its **own frequency**.

### 5.1 Pairing the dimensions

Given $x = (x_1, x_2, \ldots, x_d)$, group it into pairs:

$$(x_1, x_2), (x_3, x_4), \ldots, (x_{d-1}, x_d)$$

(Implementations differ slightly on whether they pair adjacent indices $(x_1,x_2)$ or "split-half" indices $(x_1, x_{d/2+1})$ — more on this in Section 7. The math is identical either way; it's just a permutation of which coordinates you call "the pair.")

### 5.2 One frequency per pair

Each pair $i$ (for $i = 0, 1, \ldots, d/2-1$) gets its own rotation frequency:

$$\theta_i = \text{base}^{-2i/d}$$

where `base` is a large constant, conventionally $10000$ (this is the same constant from the original sinusoidal embedding — RoPE reuses the idea of a geometric progression of frequencies, but applies it multiplicatively as rotation instead of additively as a signal).

- For $i = 0$: $\theta_0 = \text{base}^0 = 1$ → **highest frequency**, this pair rotates fast — a small change in position $m$ causes a large change in angle. This pair is sensitive to *local, short-range* position differences.
- For $i$ near $d/2$: $\theta_i \to \text{base}^{-1}$, a very small number → **lowest frequency**, this pair rotates slowly. It takes a huge change in $m$ to meaningfully change the angle. This pair captures *long-range, coarse* position information.

This is directly analogous to a set of clock hands: the $i=0$ pair is the second hand (changes fast, encodes fine-grained local position), and the slowest pair is like an hour hand (changes slowly, encodes coarse global position). Having $d/2$ different "clocks" running at geometrically spaced speeds is what lets the network reconstruct distance information at multiple resolutions simultaneously.

### 5.3 The full rotation matrix

For position $m$, the full RoPE transform on a $d$-dimensional vector $x$ is a **block-diagonal rotation matrix**:

$$R_{\Theta, m}^d = \begin{pmatrix} R(m\theta_0) & & & \\ & R(m\theta_1) & & \\ & & \ddots & \\ & & & R(m\theta_{d/2-1}) \end{pmatrix}$$

where each $R(m\theta_i)$ is the familiar 2×2 block:

$$R(m\theta_i) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

Applying RoPE to query and key projections:

$$q_m = R_{\Theta,m}^d\, W_Q x_m, \qquad k_n = R_{\Theta,n}^d\, W_K x_n$$

In words: project the token embedding into $Q$/$K$ as usual via the learned weight matrices $W_Q$, $W_K$, then apply this purely-geometric, parameter-free rotation that depends on the token's position. No new learnable parameters are introduced anywhere — RoPE adds zero parameters to the model.

### 5.4 Complex-number view, generalized

Equivalently, view $x \in \mathbb{R}^d$ as $d/2$ complex numbers $x^{(i)} = x_{2i} + i\,x_{2i+1}$. Then RoPE is just:

$$x_m^{(i)} \;\mapsto\; x^{(i)} \cdot e^{i m \theta_i} \qquad \text{for each pair } i = 0,\dots, d/2-1$$

This is why RoPE is so often introduced as "rotating Q and K in complex space" — every 2-dimensional chunk of the vector literally is one complex number, rotated by an angle that's the product of its position $m$ and its assigned frequency $\theta_i$.

---

## 6. Why Relative Position Falls Out of the Dot Product

Now let's prove the general (not just 2D) version of what we saw in Section 4. We want to show:

$$\langle q_m, k_n \rangle = g(q, k, m-n)$$

i.e., that the standard attention dot product between rotated query and rotated key is a function of content ($q$, $k$ before rotation) and the relative distance $m-n$ *only* — never $m$ and $n$ independently.

### 6.1 Per-pair derivation

Because $R^d_{\Theta,m}$ is block-diagonal, the full dot product decomposes into a sum over the independent 2D blocks:

$$q_m^\top k_n = \sum_{i=0}^{d/2-1} \left( R(m\theta_i)\, q^{(i)} \right)^\top \left( R(n\theta_i)\, k^{(i)} \right)$$

For each block, use the defining property of rotation matrices: $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$ (rotation matrices are orthogonal, so $R(\alpha)^\top = R(-\alpha)$, and rotations compose by adding angles: $R(-\alpha)R(\beta) = R(\beta-\alpha)$). Therefore:

$$\left( R(m\theta_i)\, q^{(i)} \right)^\top \left( R(n\theta_i)\, k^{(i)} \right) = q^{(i)\top} R(m\theta_i)^\top R(n\theta_i)\, k^{(i)} = q^{(i)\top} R\big((n-m)\theta_i\big)\, k^{(i)}$$

So each pair's contribution to the dot product depends **only on $(n-m)\theta_i$**, the relative position scaled by that pair's frequency, never on $m$ or $n$ individually. Summing over all pairs:

$$q_m^\top k_n = \sum_{i=0}^{d/2-1} q^{(i)\top} R\big((n-m)\theta_i\big)\, k^{(i)} = g(q, k, m-n)$$

This completes the proof for the general $d$-dimensional case. **Absolute position was injected into $q$ and $k$ individually (via $m$ and $n$), but it cancels out perfectly in the inner product, leaving only the relative offset $m-n$.** This is exactly the relative-position-encoding property we wanted from Section 2.2 — except we got it "for free" from a parameter-free, multiplicative geometric operation, not from a learned bias table.

### 6.2 Why this matters in practice

1. **Translation invariance:** the model's attention pattern between "word 5 attends to word 3" and "word 105 attends to word 103" is mathematically identical (same relative offset $-2$), regardless of absolute position. This matches our intuition about language: relationships like "verb right after subject" should behave the same wherever in the sequence they occur.
2. **KV-cache friendliness:** because rotation is applied independently per-token at the time $Q$/$K$ are computed (not as a function of the whole sequence), cached keys from earlier generation steps remain valid — you don't need to recompute the entire $K$ cache when new tokens are appended. This is part of why RoPE plays so nicely with autoregressive generation and PagedAttention-style KV caching.
3. **No extra parameters, no extra table lookup:** unlike T5-style relative bias (which needs a learned bucket table) RoPE is just trigonometric functions applied to existing $Q$/$K$ values.

---

## 7. Practical Implementation (the "rotate-half" trick)

Real implementations (LLaMA, GPT-NeoX, HF `transformers`) don't literally construct block-diagonal matrices — that would be wasteful. Instead they exploit the fact that the 2×2 rotation

$$\begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{pmatrix}x_1\cos\alpha - x_2\sin\alpha \\ x_1\sin\alpha + x_2\cos\alpha\end{pmatrix}$$

can be computed with plain elementwise multiplies and one "rotate-half" permutation, with no matrix multiply at all:

```python
def rotate_half(x):
    # split last dim in half: x = [x1, x2] (each of size d/2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin: shape (seq_len, d), precomputed per position
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

where `cos` and `sin` are precomputed once per position as:

```python
inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))   # shape (d/2,)
freqs = torch.outer(positions, inv_freq)                          # shape (seq_len, d/2)
freqs = torch.cat((freqs, freqs), dim=-1)                         # shape (seq_len, d)  (split-half pairing)
cos, sin = freqs.cos(), freqs.sin()
```

Note this uses the **"split-half"** pairing convention ($x_i$ pairs with $x_{i+d/2}$) rather than the "adjacent" convention ($x_{2i}$ pairs with $x_{2i+1}$) from Section 5.1 — it's mathematically equivalent (just a relabeling of which coordinates form each complex pair) but is what virtually every modern open-source implementation (LLaMA, Mistral, Qwen, GPT-NeoX) actually does, because it vectorizes much better on GPU than interleaved indexing.

**Key implementation facts:**

- $\cos$/$\sin$ tables only need to be computed once per sequence length and can be cached/reused.
- RoPE is applied to $Q$ and $K$ only — never to $V$. $V$ carries pure content; position only needs to influence *where attention looks*, not *what gets aggregated*.
- It's applied **after** the linear projections $W_Q, W_K$, not before — i.e., not at the embedding layer, but freshly inside every attention layer/head.

---

## 8. The Long-Term Decay Property

This is a property RoPE gets "for free" as a side effect of its construction, and it's a major reason RoPE-based models perform well: **the attention contribution between two tokens trends toward zero as their distance increases, even though the network never explicitly learned a decay function.**

### 8.1 Where the decay comes from

Recall from Section 6.1 that the full dot product is:

$$q_m^\top k_n = \sum_{i=0}^{d/2-1} q^{(i)\top} R\big((n-m)\theta_i\big)\, k^{(i)}$$

If we treat the pairs $q^{(i)}, k^{(i)}$ as roughly random/uncorrelated in direction (a reasonable approximation pre-training, and empirically a good one for trained models too), this sum behaves like a sum of $d/2$ oscillating terms — each term oscillates as $\cos\big((n-m)\theta_i + \phi_i\big)$ for some phase $\phi_i$ depending on $q^{(i)}, k^{(i)}$. Su et al. (the RoPE authors) showed that if you bound the magnitude of this sum using an Abel-summation argument, you get an upper bound on $|q_m^\top k_n|$ that **decreases as $|m-n|$ grows**, because the high-frequency pairs (small $i$, large $\theta_i$) oscillate rapidly and tend to cancel each other out in the sum as distance grows, while only the low-frequency pairs (which decay more slowly) keep contributing meaningfully at long range.

### 8.2 Intuition without the heavy math

Think of each of the $d/2$ frequency pairs as a sinusoid with a different period. When you add together many sinusoids of different (incommensurate) frequencies and look at their sum at increasing "time" offsets, the high-frequency components churn through many cycles and tend to average out (destructive interference), while only the slowest components stay roughly "in phase" for a long while. The net effect: as the relative distance $|m-n|$ increases, more and more of the $d/2$ frequency channels have "decohered" and stopped contributing constructively, so the *expected magnitude* of the dot product trends downward.

This gives RoPE an automatic, **built-in recency bias**: nearby tokens tend to get higher raw attention scores (before any learning happens) purely from the geometry of the rotation, and distant tokens' contributions are naturally damped. This is a useful inductive bias because language usually *is* locally structured (nearby words tend to be more syntactically/semantically related), and it gives the model "for free" something that ALiBi achieves via an explicit, hand-designed linear penalty — RoPE achieves a qualitatively similar effect implicitly, through interference of rotating vectors.

### 8.3 Why this matters for extrapolation (preview)

This decay property is a double-edged sword for the topic in Section 9: it's beneficial *within* the trained context length (helps the model focus locally), but it also means that at very large $|m-n|$ — especially distances *larger than anything seen during training* — the rotation angles $m\theta_i$ for the high-frequency pairs **wrap around the unit circle many, many times** and start hitting *angle values never seen during training*, producing essentially out-of-distribution, unpredictable rotation patterns. This is the root cause of the extrapolation failure we look at next.

---

## 9. The Extrapolation Problem

### 9.1 What goes wrong beyond the trained context length

Suppose a model was trained with a maximum context length $L$ (e.g., 4096 tokens). During training, position indices $m \in [0, L)$ are the *only* values the rotation angles $m\theta_i$ ever took. The network's weights were optimized assuming attention scores look like $g(q,k,m-n)$ for $|m - n| < L$.

If at inference time you feed a longer sequence (say 16k tokens), you now need rotation angles for $m$ up to 16000. Two distinct problems occur:

1. **High-frequency pairs wrap around many extra times.** Recall $\theta_0 = 1$ (the fastest pair). At $m=16000$, the angle $m\theta_0 = 16000$ radians has wrapped around the circle $16000/(2\pi) \approx 2546$ times. The network never trained on these specific extra rotation states — even though mathematically the *angle itself* is a number it has seen before (angles are periodic, $\cos$ and $\sin$ repeat every $2\pi$), the *relative offsets* $m-n$ that the high-frequency channels are now being asked to represent are simply larger than anything seen in training, so the resulting attention score patterns are out-of-distribution relative to what the model learned to interpret.
2. **The KV statistics distribution shifts.** Empirically, attention score distributions, particularly for the lowest-frequency ("slow clock") dimensions, end up looking nothing like what the model saw in training once you push position indices far past $L$. This causes attention to become erratic, and perplexity spikes dramatically once you cross the trained length.

### 9.2 The naive fix and why it fails: simple Position Interpolation intuition

The intuitive first fix is: instead of letting positions run from $0$ to (new longer) $L'$, just **squeeze** the position indices back down so they still range over $[0, L)$ — e.g., if you want to support $4\times$ the original length, scale every position by $1/4$. This is the seed idea behind Position Interpolation (Section 10), but applied naively (uniformly to *all* frequency pairs) it has a flaw we'll get to.

### 9.3 The general goal of all extrapolation methods

Every method in Sections 10–13 is solving the same problem: **how do we let a RoPE-based model handle sequences longer than $L$ (the trained context window) without retraining from scratch (or with only minimal/cheap fine-tuning), while preserving the model's learned attention behavior as much as possible?**

They differ in *how* they rescale/reinterpret the rotation angles for the different frequency pairs.

---

## 10. Position Interpolation (PI)

**Reference:** Chen et al., 2023, "Extending Context Window of Large Language Models via Positional Interpolation."

### 10.1 The idea

Instead of letting position index $m$ range up to the new target length $L'$ (which produces out-of-distribution angles, as discussed), PI **linearly compresses** every position index by the same scale factor $s = L'/L$ before computing rotation angles:

$$f'(x, m) = f\left(x, \frac{m}{s}\right)$$

So if you want to go from $L=4096$ to $L'=16384$, you set $s=4$, and the model — when given a sequence of 16384 tokens — computes rotation angles as if those tokens were really only at "virtual" positions $0$ to $4096$ (just spaced four times closer together than real tokens would be).

### 10.2 Why this works (and its main weakness)

This guarantees every rotation angle the model ever sees, even on the new long sequence, stays *within the range it was trained on* — by construction, $m/s < L$ always. This is the precise mechanism that avoids the out-of-distribution angle problem from Section 9.1.

**The weakness:** PI compresses *all* frequency pairs uniformly, including the high-frequency ("fast clock") pairs that are responsible for distinguishing *nearby* tokens from each other. Squeezing positions means tokens that are genuinely close together (e.g., 2 tokens apart) now get rotation angles as if they were less than 1 token apart. This **blurs local/short-range positional resolution** — the model becomes worse at telling adjacent tokens apart, which can hurt quality even though it does successfully extend usable context length. PI typically still needs some fine-tuning (though much less than training from scratch) to recover quality.

This single weakness — uniformly squeezing *all* frequencies, including ones that didn't need it — is exactly what NTK-aware scaling and YaRN were designed to fix.

---

## 11. NTK-Aware Scaling

**Origin:** proposed in community research (notably a Reddit/LocalLLaMA post by user "bloc97", and "kaiokendev"'s related blog) building on the **Neural Tangent Kernel (NTK)** theoretical intuition that neural networks struggle to learn high-frequency functions from low-dimensional inputs unless those high frequencies are explicitly represented.

### 11.1 The key insight

PI's mistake is treating all $d/2$ frequency channels identically. But recall from Section 5.2: high-frequency pairs ($\theta_i$ near 1) encode **fine, local** position differences, while low-frequency pairs ($\theta_i$ near $\text{base}^{-1}$) encode **coarse, long-range** differences.

NTK-aware scaling reasons: **we don't need to touch the high-frequency pairs at all** — local relationships among nearby tokens were already learned correctly and the model has plenty of "resolution" there. The actual problem is *only* with the low-frequency pairs, which simply haven't seen large enough rotation angles during training to represent very long distances. So: **extrapolate the slow pairs more, and the fast pairs less (or not at all).**

### 11.2 The mechanism: stretching the base, not the positions

Instead of literally compressing position indices like PI does, NTK-aware scaling **changes the geometric base** of the frequency formula itself:

$$\theta_i = (\text{base} \cdot \alpha)^{-2i/d}$$

for some scale-up factor $\alpha > 1$ (chosen based on how much you want to extend the context, e.g., $\alpha \approx s^{d/(d-2)}$ for a target scale factor $s$). Increasing the base this way has a non-uniform effect across $i$:

- At $i=0$ (highest frequency): $\theta_0$ stays at $1$ regardless of base — **untouched**, exactly as desired.
- At large $i$ (lowest frequency): $\theta_i$ shrinks much further than it would under PI — these channels get effectively "spread out" to cover the new, longer range, since they were undersampled in training anyway and have headroom to spare.

This achieves an interpolation-like effect (avoids unseen-angle problems for the low-frequency dimensions) while leaving high-frequency local resolution essentially intact — directly fixing PI's main weakness. The trade-off: a single global base-stretch is a blunt instrument — it doesn't *perfectly* control how much each individual frequency channel is rescaled, it's more like changing one knob that happens to affect different channels by different amounts as a side effect of the math. **YaRN, next, refines this into something more deliberate.**

---

## 12. YaRN (Yet another RoPE extensioN)

**Reference:** Peng et al., 2023, "YaRN: Efficient Context Window Extension of Large Language Models." Used in models like Mistral, Qwen, DeepSeek, and many fine-tunes for long-context support.

YaRN combines two separate ideas: **(a)** a smarter, piecewise interpolation strategy across frequency channels ("NTK-by-parts"), and **(b)** a temperature correction to the attention softmax itself.

### 12.1 NTK-by-parts interpolation

YaRN explicitly classifies each of the $d/2$ frequency channels into one of three regimes based on its **wavelength** — the number of tokens it takes for that channel's rotation to complete one full $2\pi$ cycle:

$$\lambda_i = \frac{2\pi}{\theta_i}$$

- **High-frequency channels** (short wavelength $\lambda_i$, e.g., wavelength much smaller than the original trained context $L$): these channels already complete many full rotations within the trained length, so the model has seen their *entire range of behavior* during training many times over. **Leave these channels completely unscaled** — interpolating them would only blur fine-grained local resolution, exactly the problem PI had.
- **Low-frequency channels** (long wavelength, wavelength larger than $L$): these channels never even completed one full rotation cycle during training. **Apply full interpolation** (scale exactly like PI: divide effective position by $s$) to these, since they're the ones that genuinely need to be "taught" new, longer-range behavior, and as the slowest channels they're least sensitive to losing some local precision.
- **Middle-band channels:** **smoothly blend** between "no scaling" and "full PI-style scaling" using a ramp function, so there's no hard discontinuity between the two regimes.

Concretely, YaRN defines a ramp function $\gamma(i)$ (built from the wavelength thresholds) and computes the rescaled frequency as a smooth interpolation:

$$\theta_i' = \left(1 - \gamma(i)\right)\cdot \frac{\theta_i}{s} + \gamma(i)\cdot \theta_i$$

where $\gamma(i) \in [0,1]$ ramps from $0$ (full interpolation, low-frequency end) to $1$ (no scaling, high-frequency end), with the ramp boundaries set by two hyperparameters defining "this wavelength is definitely too short to need scaling" and "this wavelength is definitely long enough to need full scaling." Empirically, the boundary tends to fall such that the majority of dimensions (a large share, typically capturing dependencies up to a couple thousand tokens) stay close to unscaled, while only a smaller fraction of the dimensions — the slow ones, responsible for very long range distances — get pulled toward the full interpolation factor.

This three-zone ("NTK-by-parts") strategy is strictly more careful than the blunt single-base-stretch of vanilla NTK-aware scaling from Section 11: it explicitly decides, per-channel, exactly how much interpolation that channel deserves based on its wavelength relative to the trained context length, rather than letting it fall out as an indirect side-effect of stretching one global base constant.

### 12.2 Attention temperature scaling

The second YaRN ingredient addresses a separate, subtler issue: even after fixing the rotation angles, *stretching* effective positions over a longer range slightly changes the **entropy (sharpness) of the attention distribution** — empirically, attention scores after frequency rescaling tend to come out a bit "flatter"/less peaked than what the model was trained to expect, hurting performance even when the angle interpolation itself is done correctly.

YaRN compensates by multiplying the **attention logits** (i.e., $q_m \cdot k_n$, before softmax) by a scalar temperature-correction factor $t$, derived empirically/analytically as a function of the scale factor $s$:

$$t(s) = \sqrt{1 + \frac{\ln(s)}{\ln(L)}}$$

(sometimes written using the ratio of new to original max position embeddings rather than $s$ directly, but it's the same idea: a $\log$-based correction that grows slowly with how much you're extending the context). This factor is applied as a multiplicative scale to the $Q$ vectors (equivalent to scaling the softmax temperature), which restores roughly the same "sharpness" of attention the model learned during pre-training, even though the underlying position range has been stretched. Notably, this temperature term means YaRN is **not purely a positional-embedding-level change** — it has a small but real interaction with the attention computation itself, distinguishing it from PI and plain NTK-aware scaling, both of which only touch the rotation angles.

### 12.3 Why YaRN needs much less fine-tuning than PI

Because the high-frequency (locally-important) channels are left untouched and the attention-sharpness mismatch is explicitly corrected for, YaRN's "zero-shot" (no fine-tuning at all) extrapolation quality is already strong, and the small amount of fine-tuning typically applied (a few hundred steps, vastly less data than PI requires) is enough to fully recover — and sometimes exceed — original-context-length quality at the new, extended length.

---

## 13. Su-Scaled RoPE / LongRoPE ("SuMA")

This is the method behind the `"su"` / `"longrope"` `rope_scaling` type used in Microsoft's **Phi-3** model family (e.g., Phi-3-mini-128k-instruct, extending a 4k-trained model to 128k context) and named after **Su et al.**, the original authors of RoPE — hence "Su-scaled."

### 13.1 The core idea: search instead of hand-designed formula

PI, NTK-aware, and YaRN all use a **hand-designed mathematical formula** to decide how much to rescale each frequency channel. LongRoPE / Su-scaled RoPE instead treats "how much should each of the $d/2$ frequency channels be rescaled" as an **optimization problem**, and solves it with an **evolutionary search** algorithm, under the constraint that the rescale factors must be non-decreasing across frequency index (preserving the intuition that lower-frequency, longer-wavelength channels should always get at least as much rescaling as higher-frequency ones — consistent with the NTK theory motivating YaRN, but instead of deriving a closed-form ramp function, it lets a search procedure discover the per-channel factors empirically by minimizing perplexity on long-context validation data).

This produces, for each attention head dimension, a **per-channel scaling factor** rather than a single global formula — i.e., instead of one smooth ramp $\gamma(i)$, you get $d/2$ independently-searched numbers.

### 13.2 Two separate factor sets: `short_factor` and `long_factor`

A distinctive feature of Su-scaled RoPE (visible directly in the Phi-3 config/implementation) is that it stores **two full sets** of per-channel scaling factors:

- **`short_factor`**: used whenever the current sequence length is within the model's *original* trained context window (e.g., ≤ 4096 for Phi-3-mini-128k, which was originally a 4k model). This keeps short-sequence behavior essentially identical to the un-extended model.
- **`long_factor`**: used whenever the sequence length exceeds the original window. These are the (evolutionary-search-discovered) factors that stretch the low-frequency channels to support up to the new target length (e.g., 128k).

At inference time, the implementation simply checks the current max position index and switches between the two factor tables:

```python
su_factor = long_factor if max(position_ids) > original_max_position_embeddings else short_factor
inv_freq = 1.0 / (su_factor * base ** (torch.arange(0, dim, 2).float() / dim))
```

This adaptive switching means the model doesn't pay any extrapolation "tax" at all for ordinary short-context usage — it only activates the long-context rescaling once it's actually needed.

### 13.3 The extra scalar scaling factor (attention-temperature analogue)

Like YaRN's logit temperature correction, Su-scaled RoPE also applies a global multiplicative correction, computed from the ratio between new and original max position embeddings:

$$\text{scaling\_factor} = \sqrt{1 + \frac{\ln\!\left(\dfrac{L'}{L}\right)}{\ln(L)}}$$

(note: structurally similar in spirit to YaRN's $t(s)$ from Section 12.2 — both are $\log$-ratio-based square-root corrections meant to compensate for attention-sharpness drift introduced by extending the effective context range).

### 13.4 LongRoPE's second trick: progressive extension

The original LongRoPE paper (which Su-scaled RoPE is built on) also introduces a **progressive extension** strategy: rather than jumping straight from the original length (e.g., 4k) to the final target (e.g., 128k) in one fine-tuning step, the model is fine-tuned in stages (e.g., 4k → 32k → 128k), re-running the evolutionary search for the best per-channel factors at each stage. This staged approach was found to produce better long-context quality than trying to search/fine-tune for the full 32× extension in one shot.

### 13.5 How it compares conceptually to YaRN

| | YaRN | Su-scaled RoPE / LongRoPE |
|---|---|---|
| How per-channel factors are chosen | Closed-form ramp function based on wavelength thresholds | Evolutionary search, optimized against validation perplexity |
| Short-context behavior | Same formula always applied (factor → ~1 naturally for high-freq channels) | Explicit separate `short_factor` table, switched in only when needed |
| Attention correction | Log-based temperature scalar on attention logits | Very similar log-based scalar on the embeddings/frequencies |
| Extension strategy | Typically single-stage fine-tune | Often staged/progressive (e.g., 4k→32k→128k) |
| Flavor | Principled, derived from frequency/wavelength theory | Empirical, search-driven, less interpretable per-channel but can adapt to messier real-world loss landscapes |

---

## 14. Comparison Table

| Method | What it modifies | Local (short-range) resolution preserved? | Needs fine-tuning? | Core mechanism |
|---|---|---|---|---|
| **Vanilla RoPE** | — (baseline) | N/A | N/A (used as trained) | Block-diagonal rotation, $\theta_i = \text{base}^{-2i/d}$ |
| **Position Interpolation (PI)** | Position index $m \to m/s$ for *all* channels uniformly | ✗ (blurred — all channels compressed equally) | Yes, moderate amount | Linearly squeeze positions into trained range |
| **NTK-aware scaling** | Base constant $\to \text{base}\cdot\alpha$ | Mostly ✓ (high-freq channels nearly untouched, as a side-effect) | Often usable with little/no fine-tuning | Stretch frequency base so low-freq channels absorb most of the extrapolation |
| **YaRN** | Per-channel ramp $\gamma(i)$ between no-scaling and full-PI-scaling, plus attention-logit temperature correction | ✓ (explicitly preserved by design) | Minimal (few hundred steps) or zero-shot | NTK-by-parts wavelength-based ramp + temperature compensation |
| **Su-scaled RoPE / LongRoPE** | Per-channel factors found via evolutionary search; separate short/long factor tables; optional staged fine-tuning | ✓ (short_factor table keeps short-context behavior intact) | Yes, staged fine-tuning typical | Search-optimized per-channel rescaling + adaptive short/long switching + log-based scalar correction |

---

## 15. Key Takeaways

- A 2D vector and a complex number are the same object; multiplying a complex number by $e^{i\phi}$ rotates it by $\phi$ without changing its length — this single fact is the entire mechanical basis of RoPE.
- RoPE rotates $Q$ at position $m$ by angle $m\theta_i$ and $K$ at position $n$ by angle $n\theta_i$, independently for each of $d/2$ frequency pairs, where $\theta_i = \text{base}^{-2i/d}$ gives a geometric spread from fast ("local") to slow ("global") rotation channels.
- Because rotation matrices satisfy $R(\alpha)^\top R(\beta) = R(\beta-\alpha)$, the absolute positions $m, n$ cancel in the $Q\cdot K$ dot product, leaving a function of content and the *relative* offset $m-n$ only — relative position encoding achieved with zero extra parameters.
- This same construction produces an emergent **long-term decay** property: the expected magnitude of the attention score trends downward as $|m-n|$ grows, because high-frequency channels destructively interfere over long distances while only slow channels keep contributing — giving the model a built-in (not learned) recency bias.
- RoPE's reliance on rotation angles that were only ever trained within $[0, L)$ means performance degrades sharply beyond the trained context length $L$ — the **extrapolation problem**.
- **PI** fixes this crudely by uniformly compressing all positions, at the cost of local resolution. **NTK-aware scaling** improves on this by stretching the frequency base so mainly the slow channels absorb the extrapolation. **YaRN** formalizes this into an explicit per-channel wavelength-based ramp plus an attention-logit temperature correction, achieving strong extrapolation with minimal or no fine-tuning. **Su-scaled RoPE / LongRoPE** takes an empirical, search-based approach to finding per-channel factors, with explicit short/long factor tables and staged fine-tuning, as used in Phi-3's 128k-context variants.

---

## 16. References

- Su, J. et al. (2021/2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* (Original RoPE paper.)
- Vaswani, A. et al. (2017). *Attention Is All You Need.* (Original sinusoidal absolute positional encoding.)
- Chen, S. et al. (2023). *Extending Context Window of Large Language Models via Positional Interpolation.* (PI.)
- Peng, B. et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.*
- Ding, Y. et al. (2024). *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* (Basis for Su-scaled RoPE used in Phi-3.)
- EleutherAI Blog, "Extending the RoPE" — community overview of PI/NTK-aware/YaRN lineage.
- Hugging Face `transformers` source: `rope_scaling` implementations (`"linear"`, `"dynamic"`, `"yarn"`, `"su"`/`"longrope"`).