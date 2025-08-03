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

Suppressing unwanted noise acoustically in a particular region is called Active Noise Control (ANC). Typically, one or more loudspeakers emit a signal (anti-noise) that has the same amplitude but the opposite phase of the noise. Noise and anti-noise add together in the air and cancel each other out. This happens because two waves of equal amplitude travelling in phase opposition physically annihilate each other. A schematic illustration is given below.

<p align="center">
<img src="/images/anc_1.png" width="65%" height="65%">
</p>

At the green points shown in the figure the acoustic summation results in perfect silence ― they are always “zero.”

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
