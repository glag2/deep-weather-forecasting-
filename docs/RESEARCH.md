# Technology research: which neural networks to use, and which not to

Decision document. Every choice is motivated by published, cited evidence, not by
intuition. The discarded technologies are listed with the reason, because knowing what
not to use is worth as much as knowing what to use.

The imposed criterion is explicit: **proven technologies only**, nothing doubtful,
nothing that entails large compromises.

---

## 1. Our problem, in terms comparable with the literature

Before reading other people's results we need to know which column of the table concerns
us, because the conclusions of the literature change radically with the regime.

| Dimension | Us | GraphCast / Pangu / FourCastNet |
|---|---|---|
| Domain | regional, 261x401, 65 degrees of latitude | global, whole sphere |
| Lead time | 6-72 hours | 10-15 days |
| Generation | **all 9 slots in one shot** | autoregressive, 40-60 steps |
| Training data | ~2 years | 40 years |
| Hardware | **1 CPU** | hundreds of TPUs/GPUs for weeks |
| Variables | 9 surface ones | 67-221 over several levels |

Three immediate consequences, which we will use to filter the literature:

1. **We are not autoregressive.** All the literature on the stability of long rollouts,
   on climate drift and on spherical representations exists to solve a problem we **do
   not have**. Adopting those solutions would mean paying their cost without collecting
   their benefit.
2. **We are not global.** The curvature of the sphere and the singularity at the poles
   motivate GraphCast and SFNO. Our box goes from 10 to 75 degrees north: the
   convergence of the meridians exists and is already handled with the latitude
   channels.
3. **The dominant constraint is the data, not the model.** With about 2 years of data
   against 40, model capacity is not the scarce resource. This is the point on which the
   research produced the most inconvenient result, section 3.

---

## 2. The controlled comparison between backbones

**Source**: Karlbauer, Maddix et al. (AWS AI Labs, Caltech, Amazon), *Comparing and
Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and Atmospheric
Dynamics*, arXiv:2407.14129.

It is the only work that compares backbones **at equal parameter count, training
protocol and variables**. Every other paper compares its own model with other people's
models trained differently, which makes it impossible to attribute the merit to the
architecture instead of to the protocol. For an architectural decision it is the right
source.

### Main result for our regime

On **short and medium** lead times, which are ours, the paper is clear-cut:

> "Over short-to-mid-ranged lead times we observe a surprising forecast accuracy of
> ConvLSTM (the only recurrent and oldest architecture in our comparison), followed by
> SwinTransformer and FourCastNet."

And, verified **specifically on 2 metre temperature**, which is our main variable:

> "Both the results on T2m and on the ACC metric support our findings, showing the
> superiority of ConvLSTM, FourCastNet, and SwinTransformer on short-to-mid-ranged
> forecasts."

The more celebrated architectures (GraphCast, SFNO, Pangu) win on a different axis:

> "In terms of stability, explicit model designs tailored to weather forecasting are
> beneficial, e.g., Pangu-Weather, GraphCast, and Spherical FNO."

Stability concerns 365-day and 50-year rollouts. **It does not concern us.**

### Two different reasons for discarding, not to be confused

A first draft of this document discarded some technologies because they were expensive
on CPU. That is a wrong criterion: cost is measured, and if a technology really is
better you pay for it. The technologies below are divided according to the **only**
legitimate criterion, that is, whether or not they solve our problem.

### Implemented and put in comparison

All three enter the benchmark as variants of the processing block, measured on the same
protocol. The cost is recorded, not presumed.

| Technology | Why it is worth it | Evidence |
|---|---|---|
| **Windowed attention (Swin)** | Among the top three in the controlled comparison on short lead times and **specifically on T2m**. It fills the defect of pure convolution: the local receptive field does not connect distant points, while advection at 72 hours moves an air mass by hundreds of kilometres. | arXiv:2407.14129, sec. 3.2.1 and B.4 |
| **Fourier neural operator (AFNO/FourCastNet style)** | **The best overall** on synthetic data (TFNO2D, RMSE 0.0041) and in the leading group on real data at short range. It costs O(N log N) thanks to the FFT, so it is not even heavy: it is among the cheapest options. It mixes information over **the whole** domain in a single shot, something convolution does not do. | arXiv:2407.14129, Tab. 1 and Fig. 2 |
| **Convolutional recurrence (ConvLSTM)** | It is **the most accurate at short range** in the controlled comparison. The cost is cut down by applying the recurrence at the bottleneck, where the grid is already reduced by 8 times per side, that is 64 times in area. The paper reports instability beyond 4M parameters: it is a real limit to verify, not to presume. | arXiv:2407.14129, sec. 3.2.1, Tab. 1, Fig. 8 |

The FFT is needed anyway for the spectral term of the loss (section 4), so the
infrastructure is shared between loss and Fourier operator.

### Discarded because they do not solve our problem

These are not discarded for cost. They would be discarded even with unlimited hardware,
because they answer questions we have not asked.

| Technology | Reason |
|---|---|
| **Spherical representations** (SFNO, icosahedral meshes) | They exist for the curvature of the sphere and the singularity at the poles. Our box goes from 10 to 75 degrees north and the convergence of the meridians is already handled with the latitude channels. Moreover, in the same comparison, SFNO turns out to be **weaker** than FourCastNet at short range: "Surprised by the competitive results of FourCastNet and comparably poor performance of SFNO" (sec. B.3). |
| **Graph Neural Network** | The advantage of a graph is representing an **irregular mesh**. Our data is a regular grid: on a regular grid a fixed-stencil GNN is literally a convolution, only implemented more slowly and with explicit message passing. On top of this, in the controlled comparison it is the only architecture that does not converge (RMSE 0.52 against 0.005 of the best one) and it goes out of memory beyond 500k parameters. |
| **Diffusion models** (GenCast) | They solve **ensemble generation**, that is, sampling alternative futures. We already produce uncertainty in explicit and calibrated form from the probabilistic heads, and that is the form the project needs ("reliability"). It would mean replacing a direct answer with a sampled one. |
| **Pretrained foundation models** (Aurora, ClimaX) | They are the only family that would really attack our dominant constraint, data scarcity. They stay out for a structural reason and not one of speed: the checkpoints are global, with fixed variables and on pressure levels that we **have not downloaded**, so they are not applicable to our input tensor without redoing the data collection. They are worth reconsidering if the project moved to pressure levels. |

One operational detail from the paper, which we will follow: it is preferable to favour
**more layers per block** instead of more blocks with few layers.

---

## 3. The inconvenient result: enlarging the model is not enough

You asked for a **bigger and more accurate** model, spending more resources. The
research says that the first part does not imply the second, and it says so explicitly:

> "we observe that all of these model backbones **'saturate'**, i.e., none of them
> exhibit so-called neural scaling, which highlights an important direction for future
> work"

In the paper's table the saturation is visible at a glance: U-Net saturates at 1M
parameters, FourCastNet at 1M, TFNO2D at 500k. Beyond that threshold the added
parameters do not buy accuracy. In some cases they make it worse: ConvLSTM goes from
RMSE 0.009 at 1M parameters to **0.44 at 4M**, that is, almost the level of persistence.

Our current model already has **9.98 million parameters** and is trained on a few
hundred windows. We are already beyond the point where the literature observes
saturation, with a data/parameter ratio much worse than that of the papers.

**Operational consequence.** I will not enlarge the model blindly. I will measure a
curve: several configurations of increasing capacity, trained with the same protocol,
with validation error and cost per epoch. If the curve rises, we grow; if it saturates
or gets worse, enlarging would mean spending hours of CPU for a worse result, and I will
tell you that with the numbers in hand. The extra resources you authorised will be spent
where the curve says they pay off, and the most likely candidates are longer training,
six folds instead of one, and new features.

---

## 4. The most useful finding: the standard loss has a proven defect

**Source**: Subich, Husain, Separovic, Yang (Environment and Climate Change Canada),
*Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified
Spherical Harmonic Loss Function*, arXiv:2501.19374.

This result explains **exactly** the defect I had measured on our model: it captures the
phase of the diurnal cycle but underestimates its amplitude, with an error of -4.8 K at
12 UTC.

### The proof

Let `Y` be the true value and `X` the forecast, with correlation `ρ` and standard
deviation `σ_X`. The expected squared error is:

```
E[MSE] = σ_X² + 1 − 2·σ_X·ρ
```

Differentiating with respect to `σ_X` and setting to zero, the minimum is **not** at
`σ_X = 1`, but at:

```
σ_X = ρ
```

The meaning is brutal: if a phenomenon is predictable only at 70 %, the forecast that
minimises the squared error is the one that has **70 % of the real amplitude**. The
model is not making a mistake: it is doing exactly what we are rewarding it for.
Damping is the optimal strategy under MSE.

This is the phenomenon known as **double penalty**: a forecast that is correct but
displaced is punished twice, once for not having put the event where it was and once for
having put it where it was not. Predicting a flat mean avoids both punishments.

### The correction

The authors separate the **amplitude** error from the **decorrelation** error,
exploiting Parseval's theorem:

```
AMSE = Σ_k ( √PSD_k(x) − √PSD_k(y) )²  +  2·max(PSD_k(x), PSD_k(y))·(1 − Coh_k(x,y))
```

The first term punishes the wrong amplitude at every scale, the second the
decorrelation, and the two no longer compensate for each other. Relevant properties:

- it is **parameter-free**, it introduces no hyperparameters to tune;
- it is zero **if and only if** x = y, like the MSE;
- it has **the same Taylor expansion** as the MSE around the exact solution, so it does
  not change the behaviour near the optimum;
- it is a **drop-in replacement** in training.

Reported result: effective resolution of GraphCast from **1250 km to 160 km**.

### Adaptation to our case, and its limits

The paper uses spherical harmonics because it works on the sphere. We are on a regular
box, so we will use the **two-dimensional Fourier transform**: the paper explicitly
authorises the substitution, because it requires only "any decomposition (partition of
unity) that obeys Parseval's theorem", and the FFT satisfies that. In torch it is
`rfft2`, available on CPU and not very expensive.

Two limits that the paper documents and that I report because they concern us directly:

1. **On 2 metre temperature the gain in sharpness is small**, because that field is
   already little damped: it is anchored to the orography, which the model knows. The
   gain there shows up in forecast skill, not in the distribution.
2. **On precipitation the AMSE makes things worse at short lead time.** The authors give
   the reason: precipitation is localised and non-negative, so its spectral
   decomposition does not resemble the Gaussian variables for which the formula is
   derived.

Hence a precise decision: **we will apply the spectral term only to the continuous and
approximately Gaussian variables** (temperature and pressure), leaving the probabilistic
heads for rain and snow on their current likelihood, which is already calibrated and
works. Applying it everywhere would be imitating the paper instead of using it.

---

## 5. The structural defect that no loss solves

**Source**: Bonavita, *On Some Limitations of Current Machine Learning Weather
Prediction Models*, Geophysical Research Letters, 2024.

> "Forecasts from Machine Learning models have energy spectra notably different from
> those of their training reanalysis fields and Numerical Weather Prediction models.
> This results in overly smooth [forecasts]."

Independent confirmation that damping is universal in data-driven models, not an
implementation defect of ours. Useful as an honest reference point: when we report the
results, a residue of damping is expected in operational models too.

---

## 6. Final decisions

| Area | Decision | Basis |
|---|---|---|
| Backbone | Convolutional U-Net as the common frame | Leading group at short range in the controlled comparison |
| Processing block | **Four variants in measured comparison**: convolution (baseline), windowed attention, Fourier operator, convolutional recurrence | They are the three leading families on short lead times; the winner is decided on our data, not on theirs |
| Capacity | **Measured curve**, not an a priori increase | Saturation observed on all backbones; we are already at 10M parameters |
| Loss, continuous variables | Spectral amplitude term via 2-D FFT | Double penalty proven analytically; it explains our -4.8 K |
| Loss, rain and snow | Current likelihood unchanged | The paper documents a worsening of the AMSE on precipitation |
| Loss, space | Area weight by latitude + weight on Vigo di Cadore | Standard practice in the field; project requirement |
| Spherical representation | Discarded | It solves the curvature of the sphere, which we do not have; and it is weaker than FourCastNet at short range |
| Graphs | Discarded | On a regular grid they are equivalent to a slower convolution; the only non-converging architecture in the comparison |
| Diffusion | Discarded | It solves ensemble generation; we already produce explicit and calibrated uncertainty |
| Foundation models | Postponed | They require pressure levels not present in our dataset |

## 7. Ideas from language models: what transfers and what does not

### 7.0 A retraction, first of all

The first draft of this section dismissed DeepSeek-V4's CSA and HCA with the sentence
"we do not have a long sequence: we have a grid read in one go". That judgement was
formulated inside the constraint of the U-Net, where there are no tokens and the
sequence is not a parameter. Since the core of the network is `GlobalContextNet`, which
tokenises the grid, **the sequence length is a design choice of ours**: with patch 8
there are 33x51 = 1683 tokens, with patch 4 they become 66x102 = 6732, and the
query-key pairs go from 2.8 million to 45 million. The premise of the old judgement has
fallen, so the judgement has to be redone. It remains true, however, and must be said
with the same clarity, that **the gain of the sparse techniques is largely a matter of
GPU kernels**: MiniMax measures 28.4x of FLOPs saved and only 14.2x of wall-clock time
on H800. On CPU, with 1683 tokens, sparsity buys nothing; it becomes interesting only if
we go down to fine patches.

### 7.1 The mechanisms of DeepSeek-V4, separated from the name

| Mechanism | How it really works | For us |
|---|---|---|
| **CSA**, compressed sparse attention | The KV of every group of *m* tokens are fused into one with a pooling weighted by softmax and learned positional biases (sequence / m); then a *lightning indexer* with low-rank queries and ReLU scores selects the top-k blocks; MQA with shared KV, grouped output projection | Applicable **only at fine patches**. At patch 8 dense attention costs less than the selection machinery |
| **HCA**, highly compressed attention | Same pooling with much larger *m*, but dense on the compressed set: no sparsity | Transferable right away and at low cost: a second branch with coarse tokens gives an almost global context at negligible price |
| Sliding window branch | The first two layers are local window only (n_win = 128), then alternating CSA/HCA | For us the local branch already exists (convolutional branch at full resolution), so the structure is the same by another route |
| **Attention sink** | A learned logit added to the denominator of the softmax, so a head can attend to almost nothing | Transferable and almost free: a head that has nothing to say stops forcing a distribution |
| **mHC**, hyper-connections | Residual stream widened n_hc times; the B matrix is projected onto doubly stochastic matrices with 20 Sinkhorn-Knopp iterations, A and C bounded by a sigmoid, static + dynamic part with a gate at small initialisation | Applicable to any residual network, U-Net included. To be measured, not assumed |
| **Muon** | Momentum with Nesterov, hybrid Newton-Schulz orthogonalisation, rescaling by sqrt(max(n,m))·gamma, decoupled weight decay; AdamW stays on embeddings, output head, RMSNorm weights and static biases/gates of mHC | It is the cheapest test we have: it attacks the measured bottleneck, that is, the steps |
| Multi-token prediction | Emitting several future steps together | **Already done** by construction: we output all nine lead times |
| Post-training with reinforcement | Alignment and reasoning | Does not apply: there is no human preference to align, there is an observation to hit |

### 7.2 What came out afterwards, and what it changes for us

DeepSeek-V4 is from the end of April 2026: since then the literature has moved. These
are the works read, with the only question that matters: what transfers to a 261x401
grid trained on a 4-thread CPU.

**MiniMax Sparse Attention** (arXiv:2606.13392, June 2026) is the simplest and most
instructive version of the same idea: a lightweight *index branch* computes token-token
scores, aggregates them per block with a max-pool, picks the top-k blocks (k = 16,
blocks of 128) and the main branch does **exact** attention only on those. Three details
hold independently of the scale, and they are the ones we would really need:

1. **The gradient of the index must be detached.** Letting the auxiliary loss of the
   indexer flow into the body of the network produces spikes in the gradient norm and a
   worsening on short contexts, because the network learns to simplify attention to
   please the index. With `stopgrad` on the input of the indexer the problem disappears.
2. **The selection is trained with an auxiliary KL** towards the attention distribution
   of the main branch, and with a warm-up in which at the beginning both branches are
   dense.
3. **The local block must always be forcibly included**, and the block size between 32
   and 128 is irrelevant for quality: you pick the most convenient one.

**CMuon** (arXiv:2608.02502, August 2026) is the most directly useful work, because it is
Muon applied to a Diffusion Transformer, that is, to a vision network with attention
blocks, not to a language model. Its thesis: applying Muon to **fused** matrices (QKV in
a single tensor, gate+up of the MLP, AdaLN modulation) creates *subspace interference*,
because the orthogonalisation builds a single preconditioner (G^T G)^-1/2 for blocks with
different gradient statistics. The correction is trivial: split the fused matrix into its
functional sub-blocks **before** Newton-Schulz. On a 675M DiT this brings 2x over AdamW
and, above all, it keeps the advantage also at the end of training, where pure Muon
flattens out.

It concerns us directly because `GlobalBlock` uses `nn.MultiheadAttention`, which has the
**QKV fused in a single tensor `in_proj_weight`**: if we adopt Muon without splitting it,
we fall exactly into the case that the paper documents as harmful. The paper also
provides the operational numbers: quintic Newton-Schulz coefficients 3.4445 / -4.7750 /
2.0315, initial Frobenius normalisation, transposition if m > n, rescaling
0.2·sqrt(max(d_out, d_in)) chosen because it makes the RMS of the update equal to ~0.2,
that is, the typical one of AdamW, and AdamW kept on embeddings and final projections.
Honest warning: those results are at 675M parameters and batch 1024; we have 1.9M
parameters and batch 4, and the Newton-Schulz iterations on CPU cost. It has to be
measured.

The rest of the Muon family serves to avoid being naive: *Delving into Muon and Beyond*
(arXiv:2602.04669) and *The Newton-Muon Optimizer* (arXiv:2604.01472) refine the
orthogonalisation operator, while *To Use or not to Use Muon* (arXiv:2603.00742) shows
that the advantage depends on the simplicity bias of the problem and **is not
universal**. No credit on entry, then: it is adopted if it wins on our bench.

**Surface descriptors** (arXiv:2607.02824, MET Norway, July 2026) is the most important
paper of all for our measured defect, and it talks neither about attention nor about
optimisers. Adding surface descriptors to the input of a data-driven model at 2.5 km:
**-1.9% error on 2 metre temperature** over the whole domain, **-12% MAE on urban areas**
thanks to the urban fraction alone, with the largest errors concentrated on mountains
and coasts. They introduce two families of inputs:

- *surface descriptors* from the soil model: forest fraction, tree height, clay, sand,
  glacier, nature, sea, city, inland waters, maximum / minimum elevation / subgrid
  silhouette, anisotropy of the orography, x and y slopes;
- *neighbourhood topographic indices* (kernel of 12.5 km, that is 5 grid points):
  standard deviation of elevation, north-south and east-west derivatives, horizon angle,
  topographic position index, valley orientation.

The stated motivation for the neighbourhood indices is precisely our problem:
*the decoder does not connect neighbouring grid points*, so the local context has to be
supplied as an input instead of being reconstructed. And their training numbers are an
embarrassing reminder: 15,000 steps with batch 16, against our 2,560 with batch 4.

Immediate practical fallout: of their descriptors we only have `lsm` and `z`. Standard
deviation of elevation, slopes, topographic position index, silhouette and valley
orientation are **computed from `z`, which we already have**, at zero download cost; soil
type, high and low vegetation, leaf area index and the subgrid orography fields are
invariant ERA5 fields that can be downloaded. This is the modification with the highest
effect/cost ratio among all those on the list, and it is the closest to the real defect:
the 5.5% gain over persistence at 24 hours.

**Global-regional coupling.** ScaleMixer (arXiv:2603.28173) couples a pretrained global
model with a high-resolution regional network through adaptive sampling of the key
positions and cross-attention between scales; *From Global to Local* (arXiv:2607.03279)
does something cheaper: it freezes a weather foundation model and trains only lightweight
multi-scale heads **in the latent space**, obtaining a resolution jump of two orders of
magnitude without retraining the body, and showing that starting from the latent beats
super-resolution on the image. The surface descriptors paper also uses the same scheme:
frozen body, second decoder trained, cost divided by ten. Structural conclusion: **the
2026 literature does not train regional networks from scratch, it grafts them onto a
pretrained global body.** We train them from scratch on a CPU. This is probably the
deepest reason for our 5.5%, and it is not solved with an attention block: it requires
pretrained weights (Aurora, Pangu, Anemoi), so it is a decision of the user, not of the
agent, because so far the rule was `timm`/`torchvision` only.

### 7.3 Order of attack, by expected effect and not by novelty

1. **Topographic descriptors derived from `z`**, no download, documented effect on the
   defect we have, it attacks the missing physics.
2. **Muon in chunked version (CMuon)**, with AdamW on norms and output head and QKV split
   in three, it attacks the measured bottleneck, the steps.
3. **HCA branch with coarse tokens + attention sink**, almost global context at
   negligible cost, without sparsity and without dedicated kernels.
4. **mHC**, plausible but without evidence in our regime.
5. **CSA with detached indexer and auxiliary KL**, only if and when we move to fine
   patches, because before that there is nothing to save.

## 8. References

1. Karlbauer, Maddix, Ansari, Han, Gupta, Wang, Stuart, Mahoney (2024). *Comparing and
   Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and
   Atmospheric Dynamics*. arXiv:2407.14129.
2. Subich, Husain, Separovic, Yang (2025). *Fixing the Double Penalty in Data-Driven
   Weather Forecasting Through a Modified Spherical Harmonic Loss Function*.
   arXiv:2501.19374.
3. Bonavita (2024). *On Some Limitations of Current Machine Learning Weather Prediction
   Models*. Geophysical Research Letters, 10.1029/2023GL107377.
4. Adamov, Oskarsson, Denby et al. (2025). *Building Machine Learning Limited Area
   Models: Kilometer-Scale Weather Forecasting in Realistic Settings*. arXiv:2504.09340.
5. Lam et al. (2023). *Learning skillful medium-range global weather forecasting*.
   Science, 10.1126/science.adi2336.
6. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted
   Windows*. ICCV.
7. DeepSeek-AI (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context
   Intelligence*. arXiv:2606.19348.
8. Lai, Xu, Yang et al. (2026). *MiniMax Sparse Attention*. arXiv:2606.13392.
9. Chen, Sun, Yuan (2026). *CMuon: Accelerating and Stabilizing Diffusion Transformer
   Training via Chunked Momentum Orthogonalization*. arXiv:2608.02502.
10. Bakketun, Haugen, Blyverket, Nipen, Muller (2026). *Enhancing a high resolution
    data-driven weather prediction model with surface descriptors*. arXiv:2607.02824.
11. Chen, Wang, Yuan et al. (2026). *Skillful Kilometer-Scale Regional Weather Forecasting
    via Global and Regional Coupling* (ScaleMixer). arXiv:2603.28173.
12. Kamzela, Kubiak, Dobosz et al. (2026). *From Global to Local: Efficient Regional
    Weather Downscaling with Global Weather Foundation Model*. arXiv:2607.03279.
13. *Delving into Muon and Beyond: Deep Analysis and Extensions* (2026).
    arXiv:2602.04669.
14. *To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters* (2026).
    arXiv:2603.00742.
15. *The Newton-Muon Optimizer* (2026). arXiv:2604.01472.
16. Sun, Li, Zhang et al. (2025). *Efficient Attention Mechanisms for Large Language
    Models: A Survey*. arXiv:2507.19595.
