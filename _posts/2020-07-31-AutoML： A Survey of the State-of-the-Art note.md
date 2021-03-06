---
layout:		post
title:      AutoML：A Survey of the State-of-the-Art阅读笔记
subtitle:	
date:       2020-07-31
author:     xuelin
header-img: img/post-web.jpg
catalog:    true
tags:
    - AutoML
    - NAS
    - HPO
---

# AutoML: A Survey of the State-of-the-Art

[He X, Zhao K, Chu X. AutoML: A Survey of the State-of-the-Art[J]. arXiv preprint arXiv:1908.00709, 2019.](https://arxiv.org/abs/1908.00709)

### 1. 介绍

文章着重介绍了两个方面：一是自动机器学习的pipeline过程总结，主要包括数据准备（Data Preparation）、特征工程（Feature Engineering）、模型生成（Model Generation）和模型评估（Model Evaluation）；二是细致地总结和比较各种主流的神经架构搜索算法（Neural Architecture Search algorithm, NAS），现有主流的NAS算法有四种类型：随机搜索（Random Search, RS）、强化学习（Reinforcement Learning, RL）、进化算法（Evolutionary Algorithm, EA）和基于梯度下降（Gradient Descent, GD）的算法。

AutoML Pipeline 框架图
![AutoML Pipeline 框架图](/assets/15961859212238.jpg)

越来越多关于AutoML的研究在逐渐展开，其中大部分的研究主要关注神经架构搜索算法（Neural Architecture Search algorithm, NAS）。NAS算法旨在通过从预定义的搜索空间中选择和组合不同的基本组件来生成健壮且性能良好的神经架构。本文将从两个方面介绍ANS算法，一是模型结构，二是超参优化（Hyperparameter Optimization, HPO）。

### 2.数据准备（Data Preparation）

数据准备我们将其分为：数据收集（Data Collection）、数据清理（Data Cleaning）和数据增强（Data Augmentation）。

- 数据收集（Data Collection）：对于深度学习任务，我们一般首选公开的大型数据集，如CIFAR10 & CIFAR100等等，但是对于一些特定的任务，很难找到合适的数据集，这里我们有两类方法解决这个问题。一是数据合成（Data Synthesis）；二是数据搜索（Data Searching），这些都可以让你收集到合适的数据集。
- 数据清理（Data Cleaning）：在进入特征生成之前，必须预处理原始数据，因为存在许多类型的数据错误（例如，冗余，不完整或不正确的数据），被广泛使用的数据清理操作包括标准化，缩放，定量特征二值化，one-hot编码定性特征，用平均值填充缺失值等等。
- 数据增强（Data Augmentation）

![-w321](/assets/15961888798106.jpg)

### 3. 特征工程（Feature Engineering）

特征工程有三个组成部分：特征选择（Feature Selection），特征构造（Feature Construction）和特征提取（Feature Extraction）。

#### 特征选择（Feature Selection）


特征选择基本流程
![特征选择基本流程](/assets/15961890817832.jpg)

#### 特征构造（Feature Construction）

特征构造是从基本特征空间或原始数据构造新特征的过程，以帮助提高模型的预测准确性和泛化性，因此其本质是增加原始特征的代表能力。 传统上，这个过程高度依赖于人类的专业知识，常见的方法是预处理转换，这些转换也应用于数据预处理，例如标准化、规范化和特征离散化等等。

#### 特征提取（Feature Extraction）

特征提取是通过一些映射函数的降维过程，其根据一些特定度量提取信息和非冗余特征。 与特征选择不同，特征提取将改变原始特征。 特征提取的核心是映射函数，可以以多种方式实现，如PCA和LDA等等。

### 4. 模型生成（Model Generation）

特征工程之后，我们需要选择模型并设置其相应的超参数。 模型选择有两种方法：传统模型选择和NAS。前者是从许多传统的机器学习算法中选择性能最佳的模型，如支持向量机（SVM）、KNN、决策树、KMeans等。而NAS方法的目标是不借助人力选择模型。本文的重点是NAS方法，主要包括两个方面：模型结构和模型参数自动优化。

#### 模型结构（Model Structure）

- 整体结构（Entire structure）：第一种直观的方法是生成一个完整的链结构神经网络，类似于传统的神经网络结构。但是这类模型的缺点也是明显的，耗时耗力，参数搜索空间等等。
- 基于单元的结构（Cell-based structure）：为了解决Entire structure的缺点，出现了Cell-based structure，该首先构建单元格结构然后堆叠预定义数量的已发现单元格，以类似于链式结构的方式生成整个架构。
- 层次结构（Hierarchical structure）：层次结构类似于基于单元的结构，但主要区别在于生成单元的方式。对于分层结构，模型存在很多级别，每个级别具有固定数量的单元格。通过迭代地合并较低级别的单元来生成较高级别的单元。

整体结构（Entire structure）
![整体结构（Entire structure）](/assets/15961894880299.jpg)

基于单元的结构（Cell-based structure）
![基于单元的结构（Cell-based structure）](/assets/15961895120758.jpg)

层次结构（Hierarchical structure）
![层次结构（Hierarchical structure）](/assets/15961895351485.jpg)

#### 参数优化（HyperParameter Optimization）

在确定网络结构的表示之后，我们需要从大型搜索空间中找到性能最佳的架构。这样的过程可以被认为是模型中节点的操作和连接的优化过程。因此，在本文中，我们将网络体系结构的搜索过程视为超参数优化（HPO），如优learning rate和batch size等等。HPO可以被分为如下几类：

1. 网格和随机搜索（Grid & Random Search）:网格搜索将可能的超参数空间划分为规则的间隔（网格），然后为网格上的所有值训练模型，并选择性能最佳的一个，而随机搜索，顾名思义，随机选择一组超参数。
2. 强化学习（Reinforcement Learning）：基于RL的算法由两部分组成。一是控制器，它是一个RNN，用于在不同的时期生成不同的子网络；二是奖励网络，用于训练和评估生成的子网络并使用 更新RNN控制器的奖励（例如准确性）。
3. 进化算法（Evolutionary Algorithm）：进化算法（EA）是一种基于人口的通用元启发式优化算法，它从生物进化中获取灵感。与传统的优化算法（如基于微积分的方法和穷举方法）相比，进化算法是一种成熟的全局优化方法，具有很高的鲁棒性和广泛的适用性。进化算法主要由四个部分组成选择，交叉，变异和更新。
4. 贝叶斯优化（Bayesian Optimization）：贝叶斯优化（BO）是一种建立目标函数概率模型的算法，然后使用该模型选择最有希望的超参数，最后在真实目标函数上评估所选择的超参数。因此，贝叶斯优化可以通过跟踪过去的评估结果来迭代地更新概率模型。
5. 梯度下降（Gradient Descent）：上面介绍的搜索算法都可以自动生成模型体系结构，同时以离散的方式搜索超参数，但是HPO的过程被视为黑盒优化问题，这就导致模型需要迭代更新许多参数，因此也就需要大量的时间和计算资源。基于梯度下降（GD）的方法可以减少搜索超参数所花费的时间。

### 5. 模型评估（Model Evaluation）

模型生成后，我们必须评估模型的性能和效果。一种直观的方法是在训练模型收敛后，根据最终模型的输出结果评估模型好坏，但是这样的方法耗时耗力。为了解决这个问题，出现了以下几种方法：
1. Low fidelity方法从训练数据集和模型大小的角度加速模型评估，如对于图像分类任务，我们可以减少图像数量或图像分辨率，或者是通过减少模型每层使用过滤器数量进行模型训练等等。
2. 迁移学习（Transfer learning）
3. 基于代理的方法（The surrogate-based methods）是另一种强大的工具，近似于黑盒功能。通常，一旦学习获得良好的近似，找到模型的最佳配置会比直接优化模型原始昂贵目标要容易得多。
4. Early stopping最初用于防止经典机器学习中的过度拟合，现在用于通过停止预测在验证集上表现不佳的评估来加速模型评估。

### 6. NAS summary

现有主流的NAS算法有四种类型：随机搜索（Random Search, RS）、强化学习（Reinforcement Learning, RL）、进化算法（Evolutionary Algorithm, EA）和基于梯度下降（Gradient Descent, GD）的算法。

主流NAS算法性能和效果比较
![](/assets/15961862541174.jpg)

主流NAS算法在CIFAR10数据集上性能和效果比较
![主流NAS算法在CIFAR10数据集上性能和效果比较](/assets/15961864136735.jpg)

### 7. 开放性问题和未来工作

- Flexible Search Space
- Applying NAS to More Areas
- Interpretability
- Reproducibility
- Robustness
- Joint Hyperparameter and Architecture Optimization
- Complete AutoML Pipeline
- Lifelong Learning

### Reference

[marsggbo知乎](https://zhuanlan.zhihu.com/p/158162306)