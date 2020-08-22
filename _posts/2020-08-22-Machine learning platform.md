---
layout:		post
title:      基于Kubernetes和Kubeflow的机器学习平台
subtitle:	
date:       2020-08-22
author:     xuelin
header-img: img/post-web.jpg
catalog:    true
tags:
    - Kubernetes
    - Kubeflow
---

### Kubeflow上的Jupyter

利用Kubeflow，每个用户或团队都将拥有自己的命名空间，在其中轻松运行工作负载。命名空间提供强大的安全保障与资源隔离机制。利用Kubernetes资源配额功能，平台管理员能够轻松限制个人或者团队用户的资源消耗上限，以保证资源调度的公平性。

在Kubeflow部署完成之后，用户可以利用Kubeflow的中央仪表板启动notebook：

![](/assets/15980785167462.jpg)

Kubeflow的notebook管理UI：用户可以在这里查看并接入现有notebook，或者启动一个新的notebook。

在Kubeflow UI中，用户可以通过选择Jupyter预设的Docker镜像、或者导入自定义镜像的URL来轻松启动新的notebook。接下来，用户需要设置对接该notebook的CPU与GPU数量，并在notebook中添加配置与密码参数以简化对外部库及数据库的访问。

![](/assets/15980785529861.jpg)

### Pipelines

Kubeflow Pipelines UI中管道的运行时执行图：

![](/assets/15978841771238.jpg)

### 通过分布式训练加快训练速度

分布式训练已经成为谷歌内部的基本规范，同时也是TensorFlow与PyTorch等深度学习框架当中最激动人心也最具吸引力的能力之一。

谷歌当初之所以要打造Kubeflow项目，一大核心诉求就是充分利用Kubernetes以简化分布式训练流程。借助Kubernetes的自定义资源，Kubeflow得以显著降低TensorFlow与PyTorch上的分布式训练难度。用户需要首先定义一种TFJob或者PyTorch资源，如下所示。接下来，由定制化控制器负责扩展并管理所有单一进程，并通过配置实现进程之间的通信会话：

```
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
name: mnist-train
spec:
tfReplicaSpecs:
Chief:
  replicas: 1
    spec:
      containers:
        image: gcr.io/alice-dev/fairing-job/mnist
        name: tensorflow
Ps:
  replicas: 1
  template:
    spec:
      containers:
        image: gcr.io/alice-dev/fairing-job/mnist
        name: tensorflow
Worker:
  replicas: 10      
    spec:
      containers:
        image: gcr.io/alice-dev/fairing-job/mnist
        name: tensorflow
```

### Kubernetes的监控

集群信息
![](/assets/15979976258032.jpg)

namespace信息
![](/assets/15979977543347.jpg)

