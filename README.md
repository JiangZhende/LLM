# 从零训练QWEN2大模型

## 参数计算
设词表大小为V，中间状态的维度为H，前馈神经网络层的中间状态维度为$H^{'}$，有L层解码层。  
输入嵌入层：包含$V * H$个参数   。  
注意力机制：注意力机制部分包含$W^Q$，$W^K$，$W^V$，$W^O$的线性变换矩阵。  
$W^Q$：包含$H * H$个参数；  
$W^K$：包含$H * H$个参数；  
$W^V$：包含$H * H$个参数；  
$W^O$：包含$H * H$个参数。  
在注意力层需要$4 * H * H$个参数。    
前馈神经网络层：有三个线性变换层，$W^U,W^G,W^D$。  
$W^U$：包含$H * H^{'}$;  
$W^G$：包含$H * H^{'}$;  
$W^D$：包含$H^{'} * H$。  
在前馈神经网络层需要$3 * H * H^{'}$个参数。  
归一化层：QWEN的归一化层使用RMS，需要H个参数，在注意力机制前后都有一个RMS，需要$2 * H$个参数。在输出层之前也需要RMS，又一个H。  
输出层：一个线性层，将隐藏层映射到词表大小，需要$H * V$个参数。  
QWEN参数量计算公式：  
$P=2 * H * V + H + L * (4 * H^2 + 3 * H * H^{'} + 2 * H)$

以qwen2.5-0.5B为例，H=896，$H^{'}$=4864，V=151936，L=24，带入上式
2 * 896 * 151936 + 896 + 24 * （ 4 * 896^2 + 3 * 896 * 4846 + 2 * 896 ) = 662008704

## 计算量估计

计算量主要在多头注意力机制和线性变换。 
批次B， N个注意力机制，每个注意力机制维度D，序列长度T，$H=ND$. 

多头注意力计算量  
线性映射计算量。  
注意力机制有三个映射层，以$XW_Q\in (T\times H)\times (H\times H)$为例，计算量为$FLOPs=2TH^2$，三个映射层，B个批次，N个注意力机制，总计算量为$6BTH^2=6BTH^2$.   
对$softmax(\frac{QK^T}{\sqrt{D}})V$公式进行分解计算。  
$QK^T$操作计算量
$Q, K, V \in R^{B \times N \times T \times D}$，先看单个注意力机制，$QK^T$的计算量为$2\times T\times D\times T=2\times T^2\times D$，批次大小为B，$2\times B\times T^2\times D$，有N个注意力机制，总和为$2\times B\times T^2\times N\times D$.   
缩放计算$\sqrt{D}$的计算量  
$QK^T\in T\times T$，进行缩放计算每个值除以$\sqrt{D}$，有N个注意力机制，B个批次，计算量为$T^2NB$。  
softmax计算量  
$\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$。  
$S=\frac{QK^T}{\sqrt{D}}\in T\times T$，每个元素进行一次指数运算$BNT^2$；每行求和需要(T-1)次运算，总共T行，总运算量为$BNT^2$；对每个值进行一次除法，需要$BNT^2$。总计为$softmax FLOPs\approx 3\times (BNT^2)$。  
与V相乘的计算量  
$A=softmax(\frac{QK^T}{\sqrt{D}})\in {T\times T}, V\in {T\times T}$，A与V相乘需要$2\times T\times T\times D$，有B个批次，N个注意力机制，这一计算量为$2BT^2ND$。  
因此一次多头注意力计算总浮点运算量为$4BT^2ND+4BT^2N+6BNTD^2$。  
由于反向传播需要计算权重梯度和输入梯度，因此反向传播运算量大约为前向传播的两倍，整个模型中多头注意力机制的计算量为$FLOPs\approx 3(4BT^2ND+4BT^2N+6BTH^2)L$。  

前馈神经网络计算量。  
公式为$FFN(X)=\sigma (XW_U+b_1)W_D+b_2$，第一层$X\in B\times T\times H,W_U\in H\times H^{'}$，$FLOPs=2CHH^{'}$；激活的计算量为$aCH^{'}$，a一般不超过6；与偏置计算量为$cH^{'}$，第二层输出层$\sigma ({\cdot})W_D$形状为$(C\times H^{'})\times(H^{'}\times H), FLOPs=2CHH^{'}$；输出偏置计算量CH。前向计算总量为$FLOPs_{Forward}\approx 4CHH^{'}+(1+a)CH^{'}+CH$，H，$H^{'}$一般很大，主导项为$4CHH^{'}$。反向传播计算量为正向传播两倍，$8CHH^{'}$
，总计算量越为$12CHH^{'}$

总计算量公式$3L(4CHT+4CTN+6CH^2) + 12LCHH^{'}=6CL(2HT+2TN+3H^2+2CHH^{'})\approx 6CL(3H^2+2CHH^{'})$

参数量计算公式$P = 2HV + H + L(4H^2 + 3HH^{'} + 2H)\approx L(4H^2+3HH^{'})$

根据上述两式，计算量和参数量的主要部分均为$L(H^2+HH^{'})$，因此$FLOPs\approx 6CP$，如果训练过程采用了激活重计算技术，反向传播时需要额外进行一次前向传播，则总运算量为$FLOPs\approx 8CP$

## 训练时间估计

训练过程中，训练时间涉及多个部分，主要包括浮点数运算，数据读写等，其中浮点数运算是训练过程的主要部分，因此，可以根据计算量的估计公式以及GPU的浮点运算能力来大致估算训练时间。具体的估计公式如下：$训练时间\approx \frac{运算量}{GPU数量\times GPU每秒浮点运算数}$，GPU每秒浮点运算数通常是GPU理论浮点运算能力的30%到70%。以LLaMA（65B）的预训练为例，参数量$P=6.5\times 10^10$，词元数$C=1.4\times 10^{12}$，训练过程采用了激活重计算技术，运算量大致为$8CP=7.28\times 10^{23}$。预训练过程中使用了2048张A100GPU，每张A100GPU每秒最多能进行$3.12\times 10^{14}$次BF16浮点数运算。假设每张GPU每秒$2\times 10^{14}$次BF16浮点数运算。根据公式计算LLaMA（65B）使用2048张A100GPU在1.4T个词元上的训练时间大致为$1.78\times 10^{16}$秒，大约20.6天，与论文中公布的21天基本一致。

## 数据

pleisto/wikipedia-cn-20230720-filtere。25w。  中文维基百科
xuqinyang/BaiduBaike-5.63M。1.73m  中文百度百科
<!-- wangrui6/Zhihu-KOL。1.01m -->
<!-- wdndev/webnovel-chinese。601k。搜集网络上的网文小说，清洗，分割后，用于训练大语言模型，共计9000本左右，大约9B左右token。 -->
TigerResearch/pretrain_zh。16.9m  Tigerbot pretrain数据的中文部分。
包含(未压缩前) 中文书籍zh-books 12G, 中文互联网zh-webtext 25G, 中文百科zh-wiki 19G


