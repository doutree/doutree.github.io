---
layout:		post
title:      AutoML框架整理
subtitle:	
date:       2020-08-01
author:     xuelin
header-img: img/post-web.jpg
catalog:    true
tags:
    - AutoML
---
    
## 全流程解决方案
### 1. MLBox
![](/assets/15962459563154.jpg)
MLBox 是一个功能强大的自动化机器学习Python库，MLBox 的主程序包包含 3 个子包，用于自动执行以下任务：
- 预处理：用于读取和预处理数据
- 优化：用于测试和交叉验证模型
- 预测：用于预测。

[房价回归（House Prices Regression）问题上运行“MLBox”的自动 ML包](https://www.kaggle.com/axelderomblay/running-mlbox-auto-ml-package-on-house-prices)

### 2. Auto-Sklearn
![](/assets/15962463001653.jpg)
Auto-sklearn 创建了一个管道，并使用贝叶斯搜索对其进行优化。在 ML 框架的贝叶斯超参数优化中，添加了两个组件：元学习用于初始化贝叶斯优化器，从优化过程中评估配置的自动集合构造。

Auto-sklearn的主要特点是一致性和稳定性，Auto-sklearn 在中小型数据集上表现良好，但它还无法在大型数据集上产生性能最先进的现代深度学习系统。

### 3. 基于树的管道优化工具（TPOT）

![](/assets/15962464316315.jpg)

TPOT 是一个 Python 自动化机器学习工具，利用遗传算法来优化机器学习管道。TPOT 扩展了 Scikit-learn 框架，使用了自己的回归器和分类器方法。TPOT 的工作原理是探索数千条可能的管道，并为数据找到最好的一个。

TPOT的优势在于其独特的优化方法，可以提供更有效的优化流程。它还包括一个能把训练好的流程直接转换为代码的工具，这对希望能进一步调整生成模型的数据科学家来说是一个主要亮点。

![](/assets/15962464646167.jpg)

相关链接
GitHub库：https://github.com/EpistasisLab/tpot
userguide：http://epistasislab.github.io/tpot

### 4. H2O

H2O是H20.ai公司的完全开源的分布式内存机器学习平台。H20 同时支持 R 和 Python，支持最广泛使用的统计和机器学习算法，包括梯度提升（Gradient Boosting）机器、广义线性模型、深度学习模型等。

H2O包括一个自动机器学习模块，使用自己的算法来构建管道。它对特征工程方法和模型超参数采用了穷举搜索，优化了管道。

H2O自动化了一些最复杂的数据科学和机器学习工作，例如特征工程、模型验证、模型调整、模型选择和模型部署。除此之外，它还提供了自动可视化以及机器学习的解释能力（MLI）。

H2O的优势在于它能够形成大型计算机集群，这使得它在规模上有所增长。它还可在python、javascript、tableau、R和Flow（web UI）等环境中使用。

h2o-docs：http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

### 5. AutoKeras

Auto-Keras 是DATA Lab构建的一个用于自动化机器学习的开源软件库。基于Keras深度学习框架，Auto-Keras提供了自动搜索深度学习模型的体系结构和超参数的功能。

API 的设计遵循 Scikit-Learn API 的经典设计，因此使用起来非常简单。当前版本提供了在深度学习过程中自动搜索超参数的功能。

Auto-Keras 的趋势是通过使用自动神经架构搜索（NAS）算法简化ML 过程。NAS 基本上用一组自动调整模型的算法，替代了深度学习工程师/从业者。

### 6. Cloud AutoML

Cloud AutoML 是来自 Google 的一套机器学习产品，利用 Google 最先进的传输学习和神经架构搜索（NAS）技术，让具有有限的机器学习专业知识的开发人员能够训练出特定的业务需求的高质量模型。

Cloud AutoML 提供了一个简单的图形用户界面（GUI），可根据自己的数据来训练、评估、改进和部署模型。目前，该套件提供以下 AutoML 解决方案：
![](/assets/15962471825378.jpg)

### 7. TransmogrifAI
TransmogrifAI 是Salesforce的一个开源自动化机器学习库。该公司的旗舰ML平台名为爱因斯坦，也由TransmogrifAI 驱动。它是一个端到端的AutoML 库，用于 Scala 编写的结构化数据，运行在  Apache Spark 之上。在以下场景中，TransmogrifAI 特别有用：
* 快速训练高质量机器学习模型，只需极少的手动调节
* 构建模块化、可重用、强类型的机器学习工作流

文档：https://docs.transmogrif.ai/en/stable/

### 8. Auto ML 

Auto_ML是一种实用工具，旨在提高从数据中获取的信息量，且无需除数据清洗外的过多工作。

该框架使用进化网格搜索算法来完成特征处理和模型优化的繁重工作。它利用其它成熟函数库（如XGBoost、TensorFlow、Keras、LightGBM和sklearn）来提高计算速度，还宣称只需最多1毫秒来实现预测，这也是这个库的亮点。

该框架可快速洞察数据集（如特征重要性）来创建初始预测模型。

https://github.com/ClimbsRocks/auto_ml

## 局部解决方案
### 特征工程
1. Featuretools
2. Boruta-py
3. Categorical-encoding
4. Tsfresh
5. Trane
6. FeatureHub

### 超参数优化
1. Skopt
2. Hyperopt
3. Simple(x)
4. Ray.tune
5. Chocolate
6. GpFlowOpt
7. FAR-HO
8. Xcessiv
9. HORD
10. ENAS-pytorch

### 参考链接

[AutoML: The Next Wave of Machine Learning](https://heartbeat.fritz.ai/automl-the-next-wave-of-machine-learning-5494baac615f)
[Automatic Machine Learning (AutoML) Landscape Survey](https://medium.com/georgian-impact-blog/automatic-machine-learning-aml-landscape-survey-f75c3ae3bbf2)
[自动机器学习工具全景图](https://developer.aliyun.com/article/626865)
[7 个AutoML库：应对机器学习的下一波浪潮](https://www.infoq.cn/article/luFB33Zy*WrHdRQh8Mo8)