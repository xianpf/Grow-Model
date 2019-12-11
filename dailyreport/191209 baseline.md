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
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.726
    ```
  - lcation: "run/downloaded FCOS" 
- ## 原原本本的代码跑出来
  - 因为GPU显存不够，没办法跑
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01
- ## 源代码修改batch size 和 lr
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 1e-4
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.172
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.327
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.161
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.095
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.226
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.195
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.345
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr1en4_191209" 
  ------
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 1e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.383
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.283
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr1en3_191210" 
  ------
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 5e-3
  - 结果：
    ```python
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.383
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.283
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    ```
  - lcation: "run/fcos_imprv_R_50_FPN_1x/Baseline_lr5en3_191211" 

- ## 源代码修改batch size 和 lr后from scratch
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01


- ## 源代码修改batch size 和 lr后from scratch， 并把FrosenBN改为BN
  - train_batch_size = 4
  - test_batch_size = 2
  - lr = 0.01

# 最后搞一个☑️的表格

# 最重要还是要回归mask，测试它们的mask表现