---
description: 因果分析
---

# causal

### A/B实验

1、双边实验：用户侧与作者侧同时AB。

2、interleaving：针对多item（搜推）场景，多个实验结果混排，大量节省实验样本

3、多重比较修正

4、方差缩减：波动大、空转有偏差的主要原因是极端样本，需要剔除；降低实验指标波动（分层法和控制变量法、回归方程和方差分析、regression adjustment、sandwich、jacknife等方法）

5、MAB：（Multi-Arm Bandit？）自动优选，算法动态调整流量分配，最大化目标转化，降低实验成本（实验期间变动策略对用户损伤大、或一类错误代价大时不适用）

6、HTE：（Heterogeneous Treatment Effects，对应ATE）异质性处理效应，实验平台提供下钻能力，获得CATE（conditional ATE，subgroup）、ITE（individual TE，uplift+GLM+Bayesian）。问题：多重比较导致一类错误膨胀

7、时间片轮转：Switchback Design，针对不满足传统实验用户独立假设SUTVA。

8、

### Observation因果

Response model -> Uplift Model (CVR提升, 按最大化Y\_1-Y\_0分裂)

Causal Tree, 直接估计模型：直接对treatment effect进行建模 (ICDM2010, qini-score)

Meta Learner, 间接估计的一种：对response effect(target)进行建模，用treatment带来的target变化作为HTE(Heterogeneous Treatment Effect)的估计。主要方法有3种:T(wo)-Learner, S(ingle)-Learner, X-Learner，思路相对比较传统的是在监督模型的基础上去近似因果关系。

Y是实验影响的核心指标\
T是treatment，通常是0/1变量，代表样本进入实验组还是对照组，对随机AB实验\
X是Confounder，可以简单理解为未被实验干预过的用户特征，通常是高维向量\
DML(Double Machine Learning)最终估计的是theta(x)，也就是实验对不同用户核心指标的不同影响

[https://github.com/uber/causalml](https://github.com/uber/causalml)

SHAP方法 (Shapley Value, 事后因子分析)

