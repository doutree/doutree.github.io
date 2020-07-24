---
layout:		post
title:		Zero-Shot Entity Linking by Reading Entity Descriptions
subtitle:	
date:       2020-07-24
author:     doutree
header-img: img/post-web.jpg
catalog: true
tags:
    - NLP
---

# Zero-Shot Entity Linking by Reading Entity Descriptions

[Paper](https://arxiv.org/abs/1906.07348)

Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin, Honglak Lee

#### 摘要

本文提出了一项称为零击实体链接（zero-shot entity linking）的新任务，在这项任务中，必须将mention链接链接到没有域内标记数据的未见实体，他们还提出了一种称为域自适应预训练（domain adaptive pre-training，DAP）的自适应预训练策略，来解决该域与在新域中链接未见实体相关的转移问题。

#### 介绍

- 先前有关实体链接的工作通常使用功能强大的资源，例如高覆盖率别名表，结构化数据和链接频率统计信息，并且还专注于链接到通用实体数据库。本文没有使用任何此类资源，并且着重于泛化到未见特殊实体。

- 本文仅作一个弱假设，即存在“实体字典”（entity dictionaries），其中包含实体（entity）及其文本描述（test description）。

- 他们的目标是建立实体链接系统，该系统可以推广到新的领域和实体词典，他们称之为world，每个world都包含自己的mentions和documents。


- 他们使用Wikias为零击实体链接（zero-shot entity linking）任务构建了一个新的数据集，因为在Wikias中，mention和context具有丰富的文档上下文（document context）可以通过阅读理解方法（reading comprehension approaches）加以利用。

- 他们假设目标实体存在于实体字典中，并留下NIL未解决（如果知识库中没有某一mention对应的实体项，则认为该mention不可链接到当前知识库，标记为NIL）

![image-20200530154734548](/assets/image-20200530154734548.png)

#### 模型

- 他们采用两步模型进行实体链接，首先是快速候选者生成阶段，该模型使用BM25（TF-IDF的一种变体）来测量mention字符串和候选文档之间的相似度，并获取前k个实体。
- 第二步是对这些实体进行排名，使用在阅读理解和自然语言推理任务方面很强的模型来比较上下文mention（mention in context）和候选实体描述（candidate entity description），如Transformers。
- 他们使用了一个称为“Full Transformer”的变体，其中上下文mention（mention in context）和候选实体描述（candidate entity description）被连接起来并输入到模型中。通过使用Transformer对 entity description 和mention in context进行联合编码，它们可以在每一层相互关联。

- 他们表明，这种深层次的cross-attention模型优于以前方法，如“ Pool-Transformer”和“ Cand-Pool-Transformer ”，这些方法没有以类似的方式使用cross-attention。

#### DAP

- 他们还以无监督的方式对不同的数据集进行了模型预训练，以改善进一步的任务。他们回顾了以前的两种方法，并提出了一种新的方法DAP（Domain-adaptive pre-training）。

- 首先是“任务自适应预训练”（Task-adaptive pre-training），其中模型在源和目标域未标记的数据上进行预训练，目的是发现跨域通用的特征。

- 第二个是“开放语料库预训练”（Open-corpus pre-training），其中模型在大型语料库上进行了预训练，其希望能够部分捕获目标域分布。

- 第三种是他们建议的新方法，称为“域自适应预训练”（DAP，Domain-adaptive pre-training），该方法仅对目标域数据进行预训练。 DAP表示能力（representational capacity）有限，因此模型应优先考虑目标域的表示质量（quality of target domain representations）。

- 在所有这三种方法中，最后都在源域标记的数据上对该模型进行了微调，他们还将这些方法结合在一起以进一步提高性能。

#### 阅读参考资料

###### 实体链接介绍

https://zhuanlan.zhihu.com/p/81073607

###### BERT介绍

https://zhuanlan.zhihu.com/p/54356280