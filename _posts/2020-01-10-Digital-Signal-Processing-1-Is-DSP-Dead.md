---
layout: post
title: Digital Signal Processing 1 - Is Digital Signal Processing (DSP) Dead?
tags: [dsp, signal processing, 5G, machine learning, artificial intelligence]
comments: true
lang: en
ref: dsp1-dspdead
---

Emre: The whole industry seems to be shifting toward Artificial Intelligence (AI), 5G, Data Science, and Machine Learning. What does that mean for the Digital Signal Processing (DSP) crowd? Is DSP dead?

Kaan: Good question. We should think of DSP both as a discipline and as a technology. Let’s take a look at each.

In fact, let me try to explain by asking you a reverse question.

Why do we need the discipline of signal processing in the first place ― why do we process signals at all?

Emre: To model and analyse physical phenomena as signals.

Kaan: Exactly. In other words, we process signals to *understand* them through physical models.

For example, we take a Fast Fourier Transform (FFT) of a signal to analyse it in the frequency domain. Looking at a spectrum helps us detect patterns or perform filtering. Other times we compress a signal, or use multi-rate processing to change its sampling frequency. The list goes on. The point is that we try to *understand* signals by decomposing them in various domains or expressing them via mathematical series.

Now, what is the goal of Machine Learning?

Emre: As the name implies, to *learn*.

Kaan: Right, but to learn *what*?

Emre: The model hidden in the data.

Kaan: Precisely ― to learn the data-driven model without necessarily *understanding* the underlying physics.

You may wonder how that is possible, but that is the beauty of machine learning. Deep-learning methods can *learn* and classify signals using relatively simple mathematics, sometimes performing better than decades of sophisticated signal-processing techniques that try to *understand* those signals. So ML is often more effective for classification problems. But classification is only a small portion of the overall problem space; many tasks still require pure signal-processing methods.

Emre: Isn’t there a paradox here? How can machine learning learn better without understanding?

Kaan: Deep learning’s mathematical framework is built for classification from the outset ― like a special-purpose machine. Signal-processing methods, on the other hand, rely on analysing or modelling everything as linear systems, backed by heavy math theorems. Non-linear analyses exist but quickly get complicated, spawning innumerable specialised methods. Deep learning skips that analysis layer and has far fewer underlying theories. In fact, saying that researchers *still* cannot fully explain how a deep network classifies would not be unfair.

Emre: So we will still need both disciplines but in different regions of the problem space?

Kaan: Yes and no. If learning alone suffices, we can stick to machine learning. But other problems demand *understanding* the signal, so we fall back on DSP. Sometimes we need a union of both. For instance, recognising that the Expectation–Maximisation (EM) algorithm is an approximate maximum-likelihood estimator under a Gaussian-noise assumption ― what if the noise isn’t Gaussian? How do we model a new distribution?

Kaan: Exactly. In some situations we may even need to *combine* DSP and machine-learning ideas. For instance, the Expectation–Maximisation (EM) algorithm is essentially an approximate maximum-likelihood estimator under the assumption of Gaussian noise. But what if the noise is *not* Gaussian? How do we model a new distribution? What do we use when EM fails? Such questions often require solutions that draw on more than one discipline—we’ll look at concrete examples later.

Emre: What disadvantages do new disciplines like ML or deep learning have compared with classical signal processing?

Kaan: From a technology perspective, ML/AI/DL algorithms typically demand large memory footprints. While that is acceptable when you must process huge datasets—where deep learning is practically unavoidable—deploying a heavy deep network to solve a *small* problem that DSP could handle elegantly is wasteful. Knowing *when* to use which tool is a mark of craftsmanship.

Speaking of technology, the world’s reaction to this shift is another important dimension.

Let me illustrate by talking about what’s happening in universities—and what happened to Texas Instruments (TI).

The engineer Alan Gatherer—often called one of the fathers of signal processing—published an IEEE article in 2017 titled “The Death and Possible Rebirth of DSP.” He wrote that universities are no longer treating signal processing as a fundamental discipline; graduates do not understand SP topics as deeply as they once did; and both on-campus and online programmes are tilting heavily toward machine-learning education. He argued the community should accept this transformation. Universities that once built DSP labs now build CUDA and OpenGL labs and teach GPU programming. Digital-filter design, FFTs, matrix factorisation, IIR filters—once core subjects—are becoming things MATLAB does for us, covered only in specialised electives.

Industry tells a complementary story.

Until roughly 2014 TI’s multi-core DSPs competed neck-and-neck with Nvidia GPUs and FPGAs in raw compute. But around 2010 a new wave began: product engineers started migrating to *server-* or *cloud-based* solutions. Fast-forward to today and virtually every AI and 5G application is expected to be *cloud-native*.

Emre: Cloud-native?

Kaan: Meaning open-source, deployable as micro-services, running inside containers, orchestrated in the cloud, and using resources optimally. Some 5G and AI workloads are latency-sensitive and therefore push data to edge servers geographically close to the data source; those edge nodes are still part of the same cloud architecture and handle real-time inference, buffering, optimisation, and M2M tasks.

Back to our topic: giants like AT&T and Google deploy hundreds—sometimes thousands—of servers to serve combined 5G/AI workloads with low latency over huge wireless-data volumes. That shift could have benefited companies such as TI, which were active in cellular base-station hardware—*if* they had embraced server-class solutions. But TI stuck to the motto “we are an electronic-components company” and therefore missed the server wave, falling far behind. It still focuses on evaluation boards, JTAG emulators, and IDEs, which drives customers away. Unless TI adopts server technologies—PCIe accelerator cards, DPDK, virtual machines, containers—it will not catch the AI/5G wave. Such a pivot needs a management style that welcomes rapid innovation, as seen at Amazon or Tesla. Remember, millennials—people who think software-first—are making today’s system-architecture decisions and directing where education, industry, and money flow.

Another angle comes from Stanford’s John Hennessy, who notes that the slowdown of Moore’s Law and Dennard scaling pushes us toward **Domain-Specific Architectures** (DSAs) and, consequently, Domain-Specific Languages. Google’s TPU is a prime example. To excel in deep learning you now have to understand TPU architecture as thoroughly as you once had to understand a DSP chip in the 1990s.

In short, disciplines like ML, AI, deep learning, and data science certainly challenge traditional signal-processing expertise. Expecting DSP engineers to ignore them is unrealistic. Fortunately, *statistical* signal processing overlaps heavily with ML, so researchers who know both may have an edge. Meanwhile, DSP engineers must now work not only with standalone embedded systems but also with cloud- and server-based technologies.

I hope this answers your question.

<a href="https://www.freecounterstat.com" title="visitor counters"><img src="https://counter4.optistats.ovh/private/freecounterstat.php?c=cx3ac8d6kfuk49ch6bj6m322mq883cqy" border="0" title="visitor counters" alt="visitor counters"></a>
