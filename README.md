# Test1_CIFAR-100
#### 期中作业实验1
在实验1中，我们选择了CIFAR-100数据集作为基准数据集。CIFAR-100是一个常用的图像分类基准数据集，包含了100个类别的60000张32x32彩色图像。我们的目标是设计一个CNN模型，并使用该模型在CIFAR-100上进行训练和测试，以比较不同的数据增强方法对于图像分类任务的影响。  
除了基准模型外，我们还采用了三种常用的数据增强方法：cutmix，cutout和mixup。其中，cutmix通过将两个不同类别的图像进行裁剪和融合来生成新的训练样本，cutout通过随机遮挡图像的一部分来增加样本的多样性，而mixup则通过线性插值的方式合成两个不同图像的特征和标签。我们将比较这三种方法与基准模型在CIFAR-100上的性能表现，以评估它们对图像分类任务的贡献。  
此外，我们还对经过cutmix，cutout和mixup处理后的三张训练样本进行了可视化。通过展示增强后的图像，我们可以直观地观察到这些数据增强方法对图像的影响，进一步理解它们在提高模型性能方面的作用。  
实验1的baseline方法中，learning rate被设置为0.01，决定了在每次迭代中更新模型参数的步长大小；优化器被设置为随机梯度下降（Stochastic Gradient Descent，SGD）优化器，并使用了动量（momentum）和权重衰减（weight decay）；迭代次数（iteration）是通过“for epoch in range(start_epoch, start_epoch+50)”这行代码来指定的，在每个迭代中，模型进行一次前向传播、反向传播和参数更新，epoch被设置为50；损失函数（loss function）是交叉熵损失函数（CrossEntropyLoss）；评价指标采用的是分类任务中常用的准确率（accuracy）。  
实验1的cutout方法中，learning rate为默认值0.1，决定了在每次迭代中更新模型参数的步长大小；优化器被设置为随机梯度下降（Stochastic Gradient Descent，SGD）优化器；epoch被设置为50，迭代次数（iteration）可以通过将总epoch数乘以每个epoch中的迭代次数得到，此处迭代次数为迭代次数为“50*len(trainloader)”；损失函数（loss function）是交叉熵损失函数（CrossEntropyLoss）；评价指标采用的是分类任务中常用的准确率（accuracy）。  
实验1的cutmix方法中，learning rate为默认值0.01，决定了在每次迭代中更新模型参数的步长大小；优化器被设置为随机梯度下降（Stochastic Gradient Descent，SGD）优化器；epoch被设置为50，迭代次数（iteration）可以通过将总epoch数乘以每个epoch中的迭代次数得到，此处迭代次数为迭代次数为“50*len(trainloader)”；损失函数（loss function）是交叉熵损失函数（CrossEntropyLoss）；评价指标采用的是分类任务中常用的准确率（accuracy）。  
实验1的mixup方法中，learning rate为默认值0.01，决定了在每次迭代中更新模型参数的步长大小；优化器被设置为随机梯度下降（Stochastic Gradient Descent，SGD）优化器；epoch被设置为50，迭代次数（iteration）可以通过将总epoch数乘以每个epoch中的迭代次数得到，此处迭代次数为迭代次数为“50*len(trainloader)”；损失函数（loss function）是交叉熵损失函数（CrossEntropyLoss）；评价指标采用的是分类任务中常用的准确率（accuracy）。
