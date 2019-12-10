# 191209 Baseline

# 实验目的
- 跑出几个baseline来作为比较对象
- 部分帮助理解FrosenBatchNorm在干啥
- 
# 实验结果
- ## FCOS 自己报告的最好的结果和提供模型测试结果
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01
  - 结果：
- ## 原原本本的代码跑出来
  - 因为GPU显存不够，没办法跑
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01
- ## 源代码修改batch size 和 lr
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01


- ## 源代码修改batch size 和 lr后from scratch
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01


- ## 源代码修改batch size 和 lr后from scratch， 并把FrosenBN改为BN
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01

# 最后搞一个☑️的表格
