---
layout:		post
title:      Kubernetes介绍
subtitle:	
date:       2020-08-20
author:     xuelin
header-img: img/post-web.jpg
catalog:    true
tags:
    - Kubernetes
---

Kubernetes是一个完备的分布式系统支撑平台，具有完备的集群管理能力，多扩多层次的安全防护和准入机制、多租户应用支撑能力、透明的服务注册和发现机制、內建智能负载均衡器、强大的故障发现和自我修复能力、服务滚动升级和在线扩容能力、可扩展的资源自动调度机制以及多粒度的资源配额管理能力。

Kubernetes优势:
* 原生的资源隔离
* 集群化自动化管理
* 计算资源(CPU/GPU)自动调度
* 对多种分布式存储的支持
* 集成较为成熟的监控和告警

### Kubernetes的组件

![](/assets/15979701655701.jpg)

![](/assets/15979088539080.jpg)

* master 可以简单的理解为控制中心
    * etcd:分布式k-v数据库，根据配置选择是cp还是ap, k8s只有api server 和etcd通讯， 其他组件均和api server通讯。
    * api server:可以理解为etcd的前置过滤器，换一个视角，它和etcd类似于mysql和文件系统。
    * controller manager: 核心，负责将现在的状态调整为etcd上应该的状态，包含了所有的实现逻辑。
    * scheduler: 简单点说就是给一个pod找一个node。
* slave 可以简单的理解为worker
    * kubelet: 负责和master连接，注册node, listen-watch 本node的任务等。
    * kube-proxy: 用于k8s service对象。
    * 容器运行时: 除了docker，k8s还支持rkt等容器实现。

### k8s集群的运行时的大致结构
![](/assets/15979090036448.jpg)

### Kubernetes对比yarn
![](/assets/15979744776731.jpg)


### Kubernetes 资源架构图
![](/assets/15979977756740.jpg)

### Kubernetes主要功能

![](/assets/15980068019936.jpg)

1. 数据卷
Pod中容器之间共享数据，可以使用数据卷。
2. 应用程序健康检查
容器内服务可能进程堵塞无法处理请求，可以设置监控检查策略保证应用健壮性。
3. 复制应用程序实例
控制器维护着Pod副本数量，保证一个Pod或一组同类的Pod数量始终可用。
4. 弹性伸缩
根据设定的指标（CPU利用率）自动缩放Pod副本数。
5. 服务发现
使用环境变量或DNS服务插件保证容器中程序发现Pod入口访问地址。
6. 负载均衡
一组Pod副本分配一个私有的集群IP地址，负载均衡转发请求到后继容器。在集群内部其他Pod可通过这个ClusterIP访问应用。
7. 滚动更新
更新服务不中断，一次更新一个Pod，而不是同时删除整个服务。
8. 服务编排
通过文件描述部署服务，使的应用程序部署变得更高效。
9. 资源监控
Node节点组件集成cAdvisor资源收集工具，可通过Heapster汇总整个集群节点资源数据，然后存储到InfluxDB时序数据库，再由Grafana展示。
10. 提供认证和授权
支持角色访问控制（RBAC）认证授权等策略。

# References

[官网](https://kubernetes.io/zh/)