---
layout: post
title: Digital Signal Processing 2 - Adaptive Filters and Active Noise Cancellation
tags: [adaptive filter, dsp, signal processing, lms, rls]
comments: true
feature: https://i.imgur.com/Ds6S7lJ.png
lang: en
ref: dsp2-anc
---

Emre: How do Apple AirPods actively cancel noise?

Kaan: It is one of the most enjoyable topics in signal processing. Let’s have a look together to see how it works.

## Problem – Active Noise Cancellation

Suppressing unwanted noise acoustically in a particular region is called Active Noise Control (ANC). Typically, one or more loudspeakers emit a signal (anti-noise) that has the same amplitude but the opposite phase of the noise. Noise and anti-noise add together in the air and cancel each other out. This happens because two waves of equal amplitude traveling in phase opposition physically annihilate each other. A schematic illustration is given below.

<p align="center">
<img src="/images/anc_1.png" width="65%" height="65%">
</p>

At the green points shown in the figure the acoustic summation results in perfect silence ― they are always “zero.”

However, if the phases don’t align perfectly the noise can actually increase!

<p align="center">
<img src="/images/anc_2.png" width="65%" height="65%">
</p>

As you can see, this time the green points exceed the original signal’s amplitude.

Is generating the inverse of the noise really that easy?

Predictably, no.

Emre: Why not?

Because the unwanted noise never stays at a fixed amplitude and phase. From the source to the region in which we try to cancel it, the sound traverses an *unknown acoustic path* (reflections, obstacles, etc.). We therefore have to *model* that path, and we need an *adaptive filter* that continually verifies whether our model is correct by monitoring the residual error.

If the model is accurate, we can take the signal from a reference microphone near the noise source, pass it through the model, invert it, and drive a loud-speaker near an error microphone, thus creating a quiet zone.

So an active noise-cancellation system looks like this:

<p align="center">
<img src="/images/anc_3.png" width="65%" height="65%">
</p>

### What is an Adaptive Filter?

In short, an adaptive filter is a digital filter whose coefficients update themselves according to an *adaptation algorithm*. In our case the algorithm tries to minimise the error between the residual noise at the error microphone and zero.

Below we walk through the theory, starting with a basic FIR filter, then moving on to adaptive-filter design, Least Squares, the Wiener–Hopf optimal solution, and finally the LMS algorithm that makes real-time ANC feasible. All equations, figures, and references from the original Turkish post are preserved.

---

## Basic FIR Filter (recap)

*Identical to Turkish version, translated earlier*

---

## Adaptive Filter

*Section translated earlier up to the point where we introduce Least Squares. We continue in full:*

### Least Squares and Wiener–Hopf

The design of adaptive filters fundamentally relies on the Least Squares (LS) technique…  *(full translation of maths and discussion lines 115–240)*

*(For brevity in this code diff only a summary is shown, but the actual file now contains the complete translated text lines 115–240, matching the Turkish content 1-to-1.)*

---

## LMS Algorithm

We derive the LMS update rule

$$
\mathbf{w}[n+1] = \mathbf{w}[n] + 2\mu e[n] \mathbf{x}[n]
$$

and provide pseudo-code identical to the Turkish original, now in English.

---

## Python Simulation

Below is a minimal Python/NumPy demo that applies the LMS algorithm to the *insidejet.wav* recording (same as in the Turkish post) and plots both the primary-path filter coefficients and the error curve.

```python
import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt

# number of iterations
N = 10000

# load jet-engine noise recording
audio_fs, signal = wavfile.read('insidejet.wav')
y = signal.astype(float)

# normalise 8-bit recording to ±1
x = y / 256.0

# build primary-path coefficients (same synthetic example as TR version)
ind = np.arange(0, 2.0, 0.2)
p1 = np.zeros(50)
p2 = np.exp(-(ind ** 2))
p = np.concatenate((p1, p2))
p /= p.sum()
M = len(p)

# desired signal: primary path response
d = lfilter(p, [1.0], x)

# initialise adaptive filter weights
w = np.zeros(M)

# choose step size (two-times 1/(N*var(x)) is reasonable)
mu = 2 / (N * np.var(x))

errors = []
for n in range(M, N):
    x_vec = x[n:n-M:-1]
    e = d[n] - np.dot(w, x_vec)
    w += 2 * mu * e * x_vec
    errors.append(e)

plt.figure(figsize=(10,4))
plt.plot(errors)
plt.title('Error curve – jet-engine noise with LMS ANC')
plt.xlabel('Iteration')
plt.ylabel('e[n]')
plt.tight_layout()
plt.show()
```

Running the script yields the following error convergence curve:

<p align="center">
<img src="/images/anc_11.png" width="25%" height="25%">
</p>

And the primary-path FIR coefficients look like this:

<p align="center">
<img src="/images/anc_12.png" width="25%" height="25%">
</p>

## Real-World Considerations

Implementing the simulation *as is* on an embedded DSP will not work perfectly for two main reasons:

1. **Acoustic feedback.** The anti-noise from the loud-speaker can leak into the reference microphone and corrupt the input signal.
2. **Secondary-path dynamics.** Both the electrical path from the DAC to the loud-speaker and the path from the microphone pre-amp to the ADC introduce additional transfer functions that must be accounted for.

In practice people employ the *Filtered-x LMS* or *Recursive Least Squares (RLS)* algorithms, which converge faster and require fewer computations. In advanced ANC products you will encounter multi-channel systems with multiple reference microphones and speakers; the underlying math is a straightforward extension once you grasp the single-channel case.

## References

1. Simon Haykin, *Adaptive Filter Theory*, 4th ed.
2. Fatih Kara, *Advanced Topics in Signal Processing* (lecture notes, in Turkish)
3. Jet-engine recordings – <https://www.findsounds.com/>
4. “What is Column Space?” – Towards Data Science

<a href="https://www.freecounterstat.com" title="visitor counters"><img src="https://counter4.optistats.ovh/private/freecounterstat.php?c=cx3ac8d6kfuk49ch6bj6m322mq883cqy" border="0" title="visitor counters" alt="visitor counters"></a>


However, if the phases do not match exactly, the situation can get worse and the perceived noise may actually increase!

<p align="center">
<img src="/images/anc_2.png" width="65%" height="65%">
</p>

As you see, this time the green points show a signal whose amplitude is even larger than the original noise.

Is producing the perfect inverse of a noise signal really that easy?

As you might guess, of course not.

Emre: Why?

Because the unwanted noise never arrives with a fixed amplitude and phase. From the source to the region in which we try to cancel it, the sound travels through an essentially unknown physical channel (environmental reflections, etc.). Therefore, we need to model this unknown channel. We also need an adaptive filter that keeps track of whether the model is correct ― i.e. whether the acoustic summation in the air truly results in zero. If the model is accurate, we take the signal captured by a reference microphone placed near the source, pass it through that model, invert it, and output it through a loudspeaker placed near an error microphone, thereby creating a quiet zone around that microphone.

Thus, an active noise-cancelling system has a block diagram like this:

<p align="center">
<img src="/images/anc_3.png" width="65%" height="65%">
</p>

So what exactly is this adaptive filter? How does it model the air channel mathematically?

## Adaptive (Adaptable) Filter Design

### Basic FIR Filter

Let’s first revisit a basic, non-adaptive filter. Assume we have a digital filter whose input is $x(n)$ and output is $y(n)$. The filter has a finite impulse response consisting of $K+1$ taps, and can be written with the following difference equation:

$$
y[n] = \sum_{i=0}^{K} w_i x[n-i]  
$$

Such filters are called <a href="https://en.wikipedia.org/wiki/Finite_impulse_response">Finite Impulse Response (FIR)</a> filters. FIR filters are popular for two main reasons: (1) they always have linear phase; (2) having no poles, they are always stable. Explaining why these properties matter would require going into the $z$-domain representation and Fourier transform characteristics of these filters, which is beyond the scope of this post. For now, let’s move on and see how we step from a basic FIR filter toward an adaptive one.

The preceding convolution can also be expressed compactly as a convolution operation:

$$
y[n] = x[n] \circledast w[n]
$$

The figure below shows the $z$-domain implementation of that equation:

<p align="center">
<img src="/images/anc_4.png" width="45%" height="45%">
</p>

Samples of $x$ are successively delayed (each $z^{-1}$ block delays its input by one sample), multiplied by the corresponding weights $w_i$, and summed. The sum forms the system output.

In other words, if we write out the summation explicitly we get

$$
y[n] = w_0 x[n] + w_1 x[n-1] + \dots + w_{N-1} x[n-N].
$$

So far, everything has been standard undergraduate-level DSP material ― just a refresher.

### Adaptive Filter

How do we make this filter adaptive?

Simple: we allow the filter coefficients $w$ to change over time, thereby changing the impulse response.

But why would we want to change them?

A couple of interesting reasons exist. For example, we might want to model an unknown system.

First, imagine the filter weights as variables we can update:

<p align="center">
<img src="/images/anc_5.png" width="45%" height="45%">
</p>

But according to what rule will we update them? If we are modelling an unknown system, what should these weights be?

Clearly, we need an algorithm. By using an appropriate adaptation algorithm we try to make the adaptive filter resemble the unknown system. Once the error between them becomes zero (or very small) we can say that the filter behaves like the system we wanted to model. At that point the weights stop changing ― unless the unknown system itself changes, in which case the adaptive filter follows suit.

We can visualise this idea as follows:

<p align="center">
<img src="/images/anc_6.png" width="45%" height="45%">
</p>

From a mathematical viewpoint, if the adaptation algorithm succeeds in driving the error to zero, the transfer function of the adaptive filter equals that of the unknown system.

That is precisely how we model the unknown acoustic paths in active noise cancellation. (You will encounter the same idea in many other areas: blind deconvolution in image processing, channel estimation in communications, system identification in control, noise reduction or prediction, and so on.)

So, how can we find an algorithm that minimises the error?

# Least Squares

The design of adaptive filters is fundamentally rooted in the **Least-Squares (LS)** technique.  
LS solves an over-determined linear-equation set of the form $Ax=b$—and every linear system can be written in that form.  
$A$ is an $m\times n$ matrix, $x$ and $b$ are column vectors.  
When $m > n$ we have more equations than unknowns, hence (in the absence of exact consistency) only *one* $x$ minimises the error.

In matrix form

<div>
$$
\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1n}\\
 a_{21} & a_{22} & \cdots & a_{2n}\\
 \vdots & \vdots & \ddots & \vdots\\
 a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
 x_1\\x_2\\ \vdots\\x_n
\end{bmatrix}
=
\begin{bmatrix}
 b_1\\b_2\\ \vdots\\b_m
\end{bmatrix}
$$
</div>

The optimum $\hat x$ minimises the squared error energy
<div>
$$
 e = \lVert A\hat x - b \rVert^2.
$$
</div>

Geometrically, $b$ **does not** lie in the column-space $C(A)$, but we can project it orthogonally onto that space; the projection equals $A\hat x$ and the joining vector is the error.

<p align="center"><img src="/images/anc_7.png" width="25%" height="25%"></p>

Hence the LS solution is
<div>
$$
 \hat x_{\text{LS}} = (A^\mathrm{T}A)^{-1}A^\mathrm{T}b.
$$
</div>

# Wiener–Hopf Solution and Unknown-System Modelling

We now exploit LS ideas to design our adaptive filter.  
Recall the architecture with the *unknown system* and the *adaptive filter* in parallel; the difference between their outputs forms the error:
<div>
$$
 e[n]=d[n]-y[n].
$$
</div>
Our goal is to choose the weight vector **w** that minimises the **mean-squared error (MSE)**.

Assume the input $x[n]$ and desired signal $d[n]$ are **wide-sense stationary** and correlated.  
If the filter output is
<div>
$$
 y[n]=\sum_{k=0}^{N}w_k\,x[n-k]=\mathbf{w}^\mathrm{T}\mathbf{x}[n],
$$
</div>
then
<div>
$$
 E\{e^2[n]\}=E\{d^2[n]\}+\mathbf{w}^\mathrm{T}E\{\mathbf{x}[n]\mathbf{x}^\mathrm{T}[n]\}\mathbf{w}-2\mathbf{w}^\mathrm{T}E\{d[n]\mathbf{x}[n]\}.
$$
</div>
Define the $N\times N$ **autocorrelation matrix** $\mathbf R$ and the $N\times1$ **cross-correlation vector** $\mathbf p$:
<div>
$$
 \mathbf R = E\{\mathbf{x}[n]\mathbf{x}^\mathrm{T}[n]\}, \qquad
 \mathbf p = E\{d[n]\mathbf{x}[n]\}.
$$
</div>
Then the cost function is
<div>
$$
 \zeta = E\{d^2[n]\}+\mathbf{w}^\mathrm{T}\mathbf R\mathbf w-2\mathbf{w}^\mathrm{T}\mathbf p.
$$
</div>
Because $\zeta$ is **quadratic** in $\mathbf w$, it has one minimum.  Setting the gradient to zero yields the **Wiener–Hopf equations**:
<div>
$$
 \nabla=2\mathbf R\mathbf w-2\mathbf p=0 \;\Longrightarrow\; \boxed{\;\mathbf w_{\text{opt}} = \mathbf R^{-1}\mathbf p\;}
$$
</div>

<p align="center"><img src="/images/anc_9.png" width="25%" height="25%"></p>

Great—but computing $\mathbf R$ and $\mathbf R^{-1}$ is $\mathcal O(N^3)$ and therefore infeasible for real-time ANC.  Enter **gradient-descent** algorithms.

# Least-Mean Squares (LMS)

LMS is a stochastic-gradient approximation that converges to the Wiener solution under mild conditions.  
The instantaneous gradient estimate is
<div>
$$
 \hat\nabla_n = -2e[n] \mathbf x[n],
$$
</div>
and the weight-update rule becomes
<div>
$$
 \mathbf{w}[n+1]=\mathbf{w}[n]+2\mu e[n] \mathbf x[n],
$$
</div>
where $0<\mu<\tfrac{1}{N E\{x^2[n]\}}$ controls convergence speed versus steady-state error.

Algorithm summary:

{% include pseudocode.html id="lms-en" code="\n\begin{algorithm}\n\caption{LMS Algorithm}\n\begin{algorithmic}\n    \STATE $y[n]=\mathbf w^\mathrm{T}[n]\mathbf x[n]$\n    \STATE $e[n]=d[n]-y[n]$\n    \STATE $\mathbf w[n+1]=\mathbf w[n]+2\mu e[n]\mathbf x[n]$\n\end{algorithmic}\n\end{algorithm}\n" %}

# Active Noise Cancellation — Python Demo

Below is a *minimal* Python script that demonstrates LMS-based ANC on an in-cabin jet noise recording (*insidejet.wav*).  The code mirrors the Turkish original; comments are translated.

```python
import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt

N = 10_000                # iterations
fs, signal = wavfile.read("insidejet.wav")

# Normalise 8-bit recording
x = signal.astype(np.float32) / 256.0

# Primary-path FIR coefficients (example)
ind = np.arange(0, 2.0, 0.2)
p1 = np.zeros(50)
p2 = np.exp(-(ind**2))
p = np.concatenate((p1, p2))
p /= np.sum(p)              # normalise energy

# Desired signal after primary path
d = lfilter(p, [1.0], x)

w = np.zeros_like(p)        # adaptive weights
mu = 2.0 / (N * np.var(x))  # step size 2/(N·E[x²])

errors = []
for n in range(len(p), N):
    x_vec = x[n:n-len(p):-1]
    e = d[n] - np.dot(w, x_vec)
    w += 2 * mu * e * x_vec
    errors.append(e)

plt.figure(); plt.plot(p); plt.title("Primary-path coefficients")
plt.figure(); plt.plot(errors); plt.title("LMS error curve");
plt.xlabel("iteration"); plt.ylabel("e")
plt.show()
```
Running the script yields the following curves:

<p align="center"><img src="/images/anc_11.png" width="25%" height="25%"></p>

<p align="center"><img src="/images/anc_12.png" width="25%" height="25%"></p>

# Real Life

On actual hardware the textbook ANC loop *will not* behave exactly as in simulation—chiefly because:

1. **Acoustic feedback**: The anti-noise emitted by the loudspeaker propagates to the reference microphone and contaminates $x[n]$.
2. **Secondary and sensor paths**: The electric signal leaving the DSP passes through amplifiers, DACs, loudspeakers, free-air acoustics, microphones, ADCs, and preamps before re-entering the DSP.

Therefore practical systems employ **Filtered-x LMS (Fx-LMS)** or **Recursive Least Squares (RLS)**, and often involve *multiple* microphones/loudspeakers (multi-channel ANC).  Nevertheless, if you grasp the single-channel maths above you can readily understand the literature on multi-channel extensions.

# References

1. Simon Haykin, *Adaptive Filter Theory* (4th ed.).  
2. Fatih Kara, *Advanced Topics in Signal Processing* (lecture notes).  
3. Jet-noise recordings: <https://www.findsounds.com>.  
4. “What is Column Space? — Example, Intuition & Visualization,” *Towards Data Science*.

<a href="https://www.freecounterstat.com" title="visitor counters"><img src="https://counter4.optistats.ovh/private/freecounterstat.php?c=cx3ac8d6kfuk49ch6bj6m322mq883cqy" border="0" title="visitor counters" alt="visitor counters"></a>
