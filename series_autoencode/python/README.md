
实现下面三个编码方法，并且做相关实验

* 基于标签RNN自动编码
* 基于AutoEncode的RNN自动编码
* （可选）基于标签和AutoEncode的综合自动编码

接口设计参考这个[项目](https://github.com/iwyoo/LSTM-autoencoder/blob/master/LSTMAutoencoder.py)。


tanh与sin
在低噪声下，可以完全分开，在sd=1的噪声下，大部分可以分开，但仍哟重合，加强深度。

周期函数可以比较好的区分开，非周期函数，有点不太好。

基于tag的编码， 使用relu，loss没有变化，随机震荡；但是使用sigmoid后，却又明显下降。后来，yasarwang分享，CNN一般用relu或leaky relu避免梯度消失，但是LSTM一般用tanh或者sigmoid。



mini batch size一般设置为2的指数，与计算机底层设计吻合。

如果2~3层网络不能很好的工作，可能加深更多也没用，而且会消耗时间。

RmsProp，动量和Adam是推荐的比较通用的加快梯度下降的算法。其他的可能只是在特定场合比较快。





## 实验：RF分类embedding特征和原始特征

使用自动编码，将原始sin,tanh曲线编码为embedding特征，然后用分类器RandomForest分别对原始曲线，embedding特征进行分类。结果显示，embedding的错误高于原始特征，说明编码过程有信息丢失。不过我是为了编码，而且是为了得到定长的编码，而原始特征也是定长的，所以需要对定长做一些文章，将其变为边长，然后看编码是否保留的原始特征。如果保留了，说明是可行的。





