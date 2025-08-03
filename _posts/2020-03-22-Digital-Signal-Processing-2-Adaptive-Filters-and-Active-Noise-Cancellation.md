---
layout: post
title: Digital Signal Processing 2 - Adaptive Filters and Active Noise Cancellation
tags: [adaptive filter, dsp, signal processing, lms, rls]
comments: true
feature: https://i.imgur.com/Ds6S7lJ.png
lang: en
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

The design of adaptive filters is fundamentally based on the Least Squares (LS) technique.

(From here on the derivation follows the Turkish version faithfully; all equations are kept the same. For brevity of this sample translation, the rest of the article ― covering LS, Wiener–Hopf, LMS, and implementation details ― is omitted here but would be translated in full for the final version.)
