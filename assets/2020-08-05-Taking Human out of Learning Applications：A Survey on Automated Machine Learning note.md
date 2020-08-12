---
layout:		post
title:      Taking Human out of Learning Applications：A Survey on Automated Machine Learning阅读笔记
subtitle:	
date:       2020-08-05
author:     xuelin
header-img: img/post-web.jpg
catalog:    true
tags:
    - AutoML
---

# Taking Human out of Learning Applications: A Survey on Automated Machine Learning

[Yao Q, Wang M, Chen Y, et al. Taking human out of learning applications: A survey on automated machine learning[J]. arXiv preprint arXiv:1810.13306, 2018.](https://arxiv.org/abs/1810.13306)

This is a review article on AutoML from the 4Paradigm company.

### INTRODUCTION

![](/assets/15965943286913.jpg)

#### Problem Definition

#### AutoML as an Interaction of Two Fields

AutoML is the intersection of automation and machine learning:

1. Machine learning:AutoML itself is a computer program that has good generalization performance, and AutoML emphasizes on **how easy** learning tools can be used and controlled. ![](/assets/15965947032718.jpg)

2. Automation:The goal of AutoML from this perspective is to construct high-level controlling approaches over underneath learning tools so that proper configurations can be found by computer programs.![](/assets/15965952408746.jpg)

#### The Definition of AutoML

An informal and intuitive description of AutoML:
![](/assets/15965953288592.jpg)

The three goals of AutoML:

* Good performance
* Less assistance from humans
* High computational efficiency

#### Basic Framework

The process of configurations tuned by humans:

![](/assets/15965956481320.jpg)

A framework for AutoML:
![](/assets/15965957469462.jpg)

#### Taxonomies of AutoML Approaches

* “What to automate”: by problem setup
* “How to automate”: by techniques
![](/assets/15965958989920.jpg)

### PROBLEM SETTINGS

Three important questions to setup an AutoML problem are:

1. What learning process we want to focus on?
2. What learning tools can be designed and used?
3. What are resultant corresponding configurations?

#### Feature Engineering

There are no common or principled methods to create features from data, so AutoML only makes limited progress in this direction. For now, we focus on feature **enhancing methods**.

**Feature Enhancing Methods**

* Dimension reduction
* Feature generation
* Feature encoding

**Search Space**
* hyper-parameters of tools, and configuration exactly refers to these hyper-parameters
* feature to be generated and selected

#### Model Selection

Models selection contains two components, i.e., picking up some classifiers and setting their corresponding hyper-parameters. 

#### Optimization Algorithm Selection

Optimization Algorithms

![](/assets/15965967810366.jpg)

#### Full Scope

There are generally two classes of full-scope AutoML approaches.

* general case: a combination of feature engineering, model selection and algorithm selection
* NAS:search good deep network architectures that suit the learning problem

### TECHNIQUES FOR OPTIMIZER

Three important questions:

* what kind of search space can the optimizer operate on?
* what kind of feedbacks it needs?
* how many configurations it needs to generate/update before a good one can be found?

#### Search Approaches

* Simple Search Approaches
   * Grid search
   * Random search
* Derivative-Free Optimization
   * Heuristic Search
      * Particle swarm optimization (PSO)
      * Evolutionary algorithms
    * Model-Based Derivative-Free Optimization
      * Bayesianoptimization
      * Classification-based optimization (CBO)
      * Simultaneous Optimistic optimization (SOO)
    * Reinforcement Learning
*  Gradient descent
*  Greedy search

### TECHNIQUES FOR EVALUATOR

Three important questions to determine the basic technique for the evaluator:

* Can the technique provide fast evaluation?
* Can the technique provide accurate evaluation?
* What feedbacks need to be provided by the evaluator?

#### Techniques

* Direct evaluation: accurate but expensive
* Sub-sampling
* Early stop
* Parameter reusing
* Surrogate evaluator

### Future Works

* Automatically creating features from the data
* Simultaneously optimize configurations and parameters
* Properties underneath problems should be further and carefully explored in order to achieve good performance
* Figure out which type of learning problems can AutoML deal with and what assumptions can we made on these learning problems
* Clarify its generalization ability