## 说明

仓库提供了Proj4的两种实现：

1. `cells_per_block = (1, 1)`
2. `cells_per_block = (n_cell, n_cell)`

这两种实现方式得到的`HoG`特征模板的维数一样，但是归一化方式不一样。

在方式1中，由于每个`block`里面只包含一个`cell`，`skimage.feature.hog`函数会对每个`cell`单独进行归一化，每个`cell`的HoG特征为单位向量，`HoG`特征模板由多个`cell`拼接得到，其不一定为单位向量。

在方式2中，由于一个`block`包含了`HoG`模板中的所有`cell`，所以`skimage.feature.hog`函数的返回结果只有一个`block`，且为单位向量。

由于`skimage.feature.hog`函数的返回结果是以`block`为单位，并按照先行后列的顺序进行排列的，对于方式1，需要对返回结果用特定的索引进行重新排列，才能构造正确的模板参数。重新排列的时候可以使用双重`for`循环实现滑动窗口操作，但是速度太慢。这里提供了矩阵化的操作进行重排列。

实验过程中发现方式2的性能比方式1更好一些。但是用了挺长时间才写完方式1的代码，还是作为一个提交版本保存下来吧。你可以通过提交历史切换不同的版本。





## 参数设置

* 设置小的`hog_cell_size`可以提高检测性能，但是运行速度会降低很多
* `orientations`与`HoG`特征的维度相关，调整`orientations`参数时，会导致SVM模型的参数发生变化，可以适当修改SVM的正则化参数`C`



### 方式1

```python
# HoG template
feature_params = {"template_size": 36, "hog_cell_size": 3, "orientations" : 31}

# training data
num_negative_examples = 10000

# svm 1
C = 5e-2
class_weight={1:10}
## confidence threshold
threshold = 0.5
topk = 200

# svm 2
C = 5e-2
class_weight={1:10}
## confidence threshold
threshold = 0.1
topk = 200
```

**AP**：

`SVM_1`$\approx 0.723$

`SVM_2`$\approx 0.361$

（我已经调不动了，电脑扛不住了）



### 方式2

```python
# HoG template
feature_params = {"template_size": 36, "hog_cell_size": 3, "orientations" : 31}

# training data
num_negative_examples = 10000

# svm 1
C = 1.0
class_weight={1:10}
## confidence threshold
threshold = 0.5
topk = 500


# svm 2
C = 1.0
class_weight={1:10}
## confidence threshold
threshold = 0.1
topk = 500
```

**AP**：

`SVM_1`$\approx 0.761$

`SVM_2`$\approx 0.529$

（我已经调不动了，电脑扛不住了）



## 调参心得

* 设置小的`hog_cell_size`

* `orientations`参数似乎影响不太大

* 设置SVM模型中的类别权重，使得假阳性率稍高一些

* 先调整SVM模型的参数，至少在训练集上看得过去，再调整其他参数

* `SVM_2`的参数太难调了，而且`hard_negative`样本的生成太耗时了，等不起

* 我人麻了。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

  

