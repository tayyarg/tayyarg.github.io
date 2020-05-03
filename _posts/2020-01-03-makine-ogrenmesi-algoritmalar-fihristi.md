---
layout: post
title: Makine Öğrenmesi 1 - Makine Öğrenmesi Uzmanının Bilmesi Gerekenler
tags: [machine learning, algorithms]
comments: true
feature: https://i.imgur.com/Ds6S7lJ.png
---

Emre: Makine öğrenmesinin kısa bir özeti ve hangi algoritmaların hangi problemlerde kullanıldığına değinebilir misin?

Kaan: Elbette. 

Makine öğrenmesi üç ana kategoriden oluşur.

1. Gözetimsiz Öğrenme
2. Gözetimli Öğrenme
3. Pekiştirmeli (Yarı-gözetimli) Öğrenme

Bu konuda çok kaynak olduğu için detaylarına girmeyeceğim. 

Özetle makine öğrenmesi konusunda uzmanlık seviyesine ulaşmak için bu kategorilere ait aşağıdaki konularda akıcı düzeyde düşünebilmek gerekir. Bu temel bilgiler ışığında gerçek dünyada karşımıza çıkan problemlerin hangi çözüm uzayında olduğu hakkında daha net bir fikre sahip olabiliriz. İlerleyen blog yazılarında aşağıdaki konular hakkında zaman zaman yazarak tüm liste hakkında tamamıyla Türkçeleştirilmiş bir kaynak oluşturmayı hedefliyorum. 

Temel Bilgiler

- Shannon'ın Kaynak Kodlama Teoremi / <a href="https://www.amazon.com/Springer-International-Engineering-Computer-Science-ebook/dp/B000WNL86A">Source Coding Theorem</a>
- <a href="https://tayyarg.github.io/dogrusal_regresyon_probleminin_bayesci_cikarimla_cozulmesi/">Bayesçi istatistik</a>- Bayes kuralı / Bayesian statistics and learning
- Cox Aksiyomları (Bayesçi olasılık teorisinin temeli) / Cox axioms
- Bayesçi Model Karşılaştırma / Bayesian model comparison
- Bilgi Teorisi - Entropi Kestirimi / <a href="http://www.cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf">Information theory - entropy estimation</a> 
- Ders kitabı önerisi - <a href="https://www.cs.ubc.ca/~murphyk/MLbook/">Machine Learning: a Probabilistic Perspective</a>

Modeller

- Saklı Markov Modelleri / (HMMs)
- <a href="https://en.wikipedia.org/wiki/State-space_representation">Durum Uzay Modelleri</a> / State space models (SSMs)
- Boltzmann Makineleri / Boltzmann machines
- Grafik Modelleri / directed, undirected, factor graphs

Algoritmalar

- Beklenti-Maksimizasyon / The EM Algorithm
- İnanç Yayılımı / Belief propagation
- İleri-Geri / Forward-backward
- <a href="https://tayyarg.github.io/kalman-filtreleme/">Kalman Filtreleme</a> / Kalman filtering and extended Kalman filtering
- Varyasyonel Metodlar / Variational methodu
- Laplas Yakınsaması ve Bayes Bilgi Kriteri / Laplace approximation and BIC
- Markov Zinciri Monte Carlo yöntemleri / Markov Chain Monte Carlo
- Parçacık Filtreleme / Particle filtering
- Beklenti Yayılım / Expectation propagation

Gözetimsiz Öğrenme

- Faktör Analizi - boyut azaltma / principal component analysis - PCA
- Bağımsız Bileşen Analizi / independent component analysis - ICA
- Karışım Modelleriyle Kümeleme  / Mixture models clustering / k-means
- Tekil Değer Ayrışımı / SVD - singular value decomposition

Gözetimli Öğrenme 

- <a href="https://tayyarg.github.io/dogrusal_regresyon_probleminin_bayesci_cikarimla_cozulmesi/">Doğrusal regresyon</a>
- <a href="https://tayyarg.github.io/dogrusal_regresyon_probleminin_bayesci_cikarimla_cozulmesi/">Gauss Süreç Regresyonu</a> / <a href="http://www.gaussianprocess.org/gpml/">Gaussian Process Regression</a>
- Lojistik regresyon 
- Karar Ağaçları / Decision Trees
- Rastgele Orman / Random Forest
- Kolektif öğrenme metodları / Ensemble methods 
- Naif Bayes Sınıflandırması
- Tek Katmanlı Öğrenme — Perceptron
- Sinir Ağları (çok-katmanlı perceptronlar) ve geri-yayılım / backpropagation
- Destek Vektör Makineleri / Support vector machines

Pekiştirmeli Öğrenme / Reinforcement Learning

- Değer fonksiyonları / Value functions
- Bellman Denklemi
- Değer İterasyonu
- Politika İterasyonu
- Q-öğrenme
- Aktör-Kritik Algoritması
- Zamansal Fark Öğrenimi / TD(lambda)

Basit öğrenme teorisi

- Vapnik–Chervonenkis Boyutu
- Regülarizasyon

Yukarıda sözü geçen konuları işleyen çok güzel ders kitapları var. Kendi çalıştığım ya da üniversitede ders olarak okuduğum kitapları daha iyi bildiğim için linklerini ekledim. 
