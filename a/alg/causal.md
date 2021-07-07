---
description: 因果分析
---

# causal

Response model -&gt; Uplift Model \(CVR提升, 按最大化Y\_1-Y\_0分裂\)

Causal Tree, 直接估计模型：直接对treatment effect进行建模 \(ICDM2010, qini-score\)

Meta Learner, 间接估计的一种：对response effect\(target\)进行建模，用treatment带来的target变化作为HTE\(Heterogeneous Treatment Effect\)的估计。主要方法有3种:T\(wo\)-Learner, S\(ingle\)-Learner, X-Learner，思路相对比较传统的是在监督模型的基础上去近似因果关系。

Y是实验影响的核心指标  
T是treatment，通常是0/1变量，代表样本进入实验组还是对照组，对随机AB实验  
X是Confounder，可以简单理解为未被实验干预过的用户特征，通常是高维向量  
DML\(Double Machine Learning\)最终估计的是theta\(x\)，也就是实验对不同用户核心指标的不同影响

[https://github.com/uber/causalml](https://github.com/uber/causalml)

SHAP方法 \(Shapley Value, 事后因子分析\)



