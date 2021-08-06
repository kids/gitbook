# ltv&churn

\(Customer\) LifeTime Value:

Beta Geometric Negative Binomial Distribution \(BG / NBD\) 估计未来一段时间内用户的购买次数

gamma-gamma 估计未来一段时间内用户的消费金额

以上都是假设分布模型，用最大似然fit参数



Churn:

SHAP \([https://github.com/slundberg/shap](https://github.com/slundberg/shap)\)



问题 \(从预测扩展到指导实践\)：

* 频次-特征 的关系
* 频次变化模式聚类
* 突然启动用户的特征
* 间隔是否符合poisson/BG分布？参数化异常值、聚类分析
* 各种可能调节的工具特征&lt;-业务侧
* 用户关系特征？graph
* 轨迹特征？平台refill/真实需求

