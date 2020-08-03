# Mask R-CNN 原理	

2020.08.01 			李中中

何凯明和他的团队在2018年提出的Mask R-CNN是实力分割领域最具有影响力的深度学习算法之一，近年来，我们利用Mask R-CNN框架解决了许多实际的问题。正确理解Mask R-CNN对于实际优化它的参数有着重要的意义，本文将阐述Mask R-CNN的基本原理。



## 基本框架

Mask-RCNN 大体框架还是 Faster-RCNN 的框架，可以说在基础特征网络之后又加入了全连接的分割子网，由原来的两个任务（分类+回归）变为了三个任务（分类+回归+分割）。Mask R-CNN 采用和Faster R-CNN相同的两个阶段:

* 第一个阶段具有相同的第一层(即RPN)，扫描图像并生成提议（proposals，即有可能包含一个目标的区域）;

* 第二阶段，除了预测种类和bbox回归，并添加了一个全卷积网络的分支，对每个RoI预测了对应的二值掩膜(binary mask)，以说明给定像素是否是目标的一部分。所谓二进制mask，就是当像素属于目标的所有位置上时标识为1，其它位置标识为 0。

总体流程如下：

1. 首先，输入一张预处理后的照片。
2. 然后， 输入到一个预训练好的神经网络中（ResNet等）获得对应的特征映射（Feature map）
3. 接着，对这个Feature map中的每一个点设定预定的ROI（Region of Interest），从而获得多个候选的ROI。
4. 接着，对这些剩下的ROI进行ROI Align操作
5. 最后，对这些ROI进行分类、回归以及MASK生成（在每个ROI里面进行FCN操作）



<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\Mask R-CNN framework.png" alt="mask r-cnn framework" style="zoom: 50%;" />

<center>图1 Mask R-CNN 总体框图</center>

## 架构分解

![maskrcnn details](C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\Mask R-CNN details.png)

<center> 图2 Mask R-CNN展开图</center>

### 1.  图像预处理

为了便于训练，图像在输入网络前需要进行一定的预处理。

![img preprocessing](C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\imge preprocessing.jpeg)

<center> 图像处理流程 </center>

* **Subtraction of mean**：图片张量[ w, h, c ] 是输入图像需要减去所训练样本和验证样本张量的平均值，使得每个维度的平均值为0.
* **Rescale**： 统一输入尺寸大小为1024*1024， 若长宽不是1:1， 那么将长的一侧统一到1024，窄的一侧按比例伸缩。
* **Padding**： 将长度不足的一侧进行填充，填充值为0。将图片将当涉及到特征金字塔网络(FPN)时，这是必要的(在下一节中解释)。所有的填充都只在最右边缘和最底边缘完成，因此不需要改变目标的坐标，因为坐标系是从最左上角开始的。

**注意: 用于生成锚点和过滤步骤的图像高度和宽度将作为缩放后的图像，而不是填充后的图像。**



### 2. Feature pyramid networks (FPN) backbone——Part 1

#### 2.1 backbone

backbone是一系列的卷积层用于提取图像的feature maps，比如可以是VGG16，VGG19，GooLeNet，ResNet50，ResNet101等，这里主要讲解的是ResNet101的结构。

ResNet（深度残差网络）实际上就是为了能够训练更加深层的网络提供了有利的思路，毕竟之前一段时间里面一直相信深度学习中网络越深得到的效果会更加的好，但是在构建了太深层之后又会使得网络退化。ResNet使用了跨层连接，使得训练更加容易。

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\resNet.png" style="zoom:60%;" />

<center> 残差网络结构 </center>

网络试图让一个block的输出为f(x) + x，其中的f(x)为残差，当网络特别深的时候残差f(x)会趋近于0，从而f(x) + x就等于了x，即实现了恒等变换，不管训练多深性能起码不会变差。

在网络中只存在两种类型的block，在构建ResNet中一直是这两种block在交替或者循环的使用，所有接下来介绍一下这两种类型的block**（indetity block， conv block）：**



* **Indetity** **Block**

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\identityblock.png" style="zoom:50%;" />

 <center> 跳过三个卷积的identity block </center>

​		图中可以看出该block中直接把开端的x接入到第三个卷积层的输出，所以该x也被称为shortcut，相当于捷径似得。注意主路上第三个卷积层使用激活层，在相加之后才进行了ReLU的激活。

* **Conv Block**

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\convblock.png" style="zoom:50%;" />

<center>跳过三个卷积并在shortcut上存在的卷积的conv block</center>

​		与identity block其实是差不多的，只是在shortcut上加了一个卷积层再进行相加。注意主路上的第三个卷积层和shortcut上的卷积层都没激活，而是先相加再进行激活的。



其实在作者的代码中，主路中的第一个和第三个卷积都是1*1的卷积（改变的只有feature maps的通道大小，不改变长和宽），为了降维从而实现卷积运算的加速；注意需要保持shortcut和主路最后一个卷积层的channel要相同才能够进行相加。下面展示Res-net101的完整框架：

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\resnet101.png" style="zoom:67%;" />

从图中可以得知ResNet分为了5个stage，C1-C5分别为每个Stage的输出，这些输出在后面的FPN中会使用到。你可以数数，看看是不是总共101层，数的时候除去BatchNorm层。注：stage4中是由一个conv_block和22个identity_block，如果要改成ResNet50网络的话只需要调整为5个identity_block.



#### 2.2 Feature Pyramid Map

##### 2.2.1 概念

FPN的提出是为了实现更好的feature maps融合，一般的网络都是直接使用最后一层的feature maps，虽然最后一层的feature maps 语义强，但是位置和分辨率都比较低，容易检测不到比较小的物体。FPN的功能就是融合了底层到高层的feature maps ，从而充分的利用了提取到的各个阶段的特征（ResNet中的C2-C5 ）。**简单来说，就是把底层的特征和高层的特征进行融合，便于细致检测**。

FPN是为了**自然地利用CNN层级特征的金字塔形式，同时生成在所有尺度上都具有强语义信息的特征金字塔**。所以FPN的结构设计了**top-down结构和横向连接，以此融合具有高分辨率的浅层layer和具有丰富语义信息的深层layer。\**这样就实现了\**从单尺度的单张输入图像，快速构建在所有尺度上都具有强语义信息的特征金字塔，同时不产生明显的代价。**

如下图所示： **Top**: 一个带有skip connection的网络结构在预测的时候是在finest level（自顶向下的最后一层）进行的，简单讲就是经过多次上采样并融合特征到最后一步，拿最后一步生成的特征做预测。**Bottom**: FPN网络结构和上面的类似，区别在于预测是在每一层中独立进行的。后面的实验证明finest level的效果不如FPN好，原因在于FPN网络是一个窗口大小固定的滑动窗口检测器，因此在金字塔的不同层滑动可以增加其对尺度变化的鲁棒性。另外虽然finest level有更多的anchor，但仍然效果不如FPN好，说明增加anchor的数量并不能有效提高准确率。

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\FPN net.png" style="zoom: 33%;" />

##### 2.2.2 特征融合图

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\FPN img.jpeg" style="zoom: 67%;" />

<center> 特征金字塔特征提取过程</center>

特征金字塔提取图像特征可以分为两个部分： 自下而上（**down-to-top pathway**）和自上而下（**top-to-down pathway**）两条路径，或者将自下而上路径称为编码过程，自上而下路径称为解码过程。

* **Down-to-top pathway**

  CNN的前馈计算就是自下而上的路径，特征图经过卷积核计算，通常是越变越小的，也有一些特征层的输出和原来大小一样，称为“相同网络阶段”（same network stage ）。对于本文的特征金字塔，作者为每个阶段定义一个金字塔级别， 然后选择每个阶段的最后一层的输出作为特征图的参考集。 这种选择是很自然的，因为每个阶段的最深层应该具有最强的特征。具体来说，对于ResNets，作者使用了每个阶段的最后一个残差结构的特征激活输出。将这些残差模块输出表示为{C2, C3, C4, C5}，对应于conv2，conv3，conv4和conv5的输出，并且注意它们相对于输入图像具有{4, 8, 16, 32}像素的步长(也就是感受野)。考虑到内存占用，没有将conv1包含在金字塔中。

* **Top-to-down pathway**

  自上而下的路径（**the top-down pathway** ）是如何去结合低层高分辨率的特征呢？方法就是，**把更抽象，语义更强的高层特征图进行上取样，然后把该特征横向连接（lateral connections ）至前一层特征，因此高层特征得到加强**。值得注意的是，横向连接的两层特征在空间尺寸上要相同。这样做应该主要是为了利用底层的定位细节信息。

  下图显示连接细节。把高层特征做2倍上采样（最邻近上采样法，可以参考反卷积），然后将其和对应的前一层特征结合（前一层要经过1 * 1的卷积核才能用，目的是改变channels，应该是要和后一层的channels相同），结合方式就是做**像素间的加法**。重复迭代该过程，直至生成最精细的特征图。迭代开始阶段，作者在C5层后面加了一个1 * 1的卷积核来产生最粗略的特征图，最后，作者用3 * 3的卷积核去处理已经融合的特征图（为了消除上采样的混叠效应），以生成最后需要的特征图。**为了后面的应用能够在所有层级共享分类层，这里作者固定了3\*3卷积后的输出通道为d,这里设为256.因此所有额外的卷积层（比如P2）具有256通道输出。这些额外层没有用非线性。**

  {C2, C3, C4, C5}层对应的融合特征层为{P2, P3, P4, P5}，对应的层空间尺寸是相通的。

  <img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\detail_upsampling.png" style="zoom:50%;" />

从图中可以看出+的意义为：左边的底层特征层通过1×1的卷积得到与上一层特征层相同的通道数；上层的特征层通过上采样得到与下一层特征层一样的长和宽再进行相加，从而得到了一个融合好的新的特征层。举个例子说就是：C4层经过1×1卷积得到与P5相同的通道，P5经过上采样后得到与C4相同的长和宽，最终两者进行相加，得到了融合层P4，其他的以此类推。

**注：P2-P5是将来用于预测物体的bbox，box-regression，mask的，而P2-P6是用于训练RPN的，即P6只用于RPN网络中。**



#### 2.3 输出

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\feature_map_output.jpg" style="zoom:87%;" />

经过高层特征与底层特征的融合之后，每层进行特征输出，分别得到[P2, P3, P4, P5, P6]五层。其中RPN网络将用到[P2, P3, P4, P5, P6]， 而目标检测网络部分用到[P2, P3, P4, P5]，也就是说，P6 16×16的特征张量只用于RPN（感兴趣区域生成网络中），用途后面具体再聊。

* rpn_feature_maps: [P2, P3, P4, P5, P6]
* mrcnn_feature_maps: [P2, P3, P4, P5]

| 特征层 | 形状          |
| ------ | ------------- |
| P2     | 256*256 @ 256 |
| P3     | 128*128 @ 256 |
| P4     | 64*64 @ 256   |
| P5     | 32*32 @ 256   |
| P6*    | 16*16 @ 256   |



### 3 Proposal Region Of Interest ——Part 2

第二部分主要介绍感兴趣区域（ROI）的生成规则，他主要有三个部分组成：锚框（anchor）生成、区域提议网络（Region Proposal Network，RPN）、区域生成网络（Proposal Layer）。如下图所示，最左侧的**anchors**是生成锚框流程，中间的**build_rpn_model->rpn_graph**是构建RPN网络流程，最右侧的**ProposalLayer**是筛选ROIs的生成建议框流程。



<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\rpn_network.jpg" style="zoom: 67%;" />

<center> 特征金字塔特征提取过程</center>

#### 3.1 锚框生成

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\anchors_f.png" style="zoom: 33%;" />

首先，需要理解锚框的概念，在这篇[博文](https://zhuanlan.zhihu.com/p/63024247)中对锚框有着很好的说明。笼统的说，P2~P6的特征图相当于将原来1024×1024的图像进行了分块，将每一块都压缩成一个像素点，然后，以该像素点为中心进行框选周边区域，为了检测多种形状的物体，同一个像素区域利用多个锚框进行框选。具体参数如下：

- 配置参数

- - **RPN_ANCHOR_SCALES**是anchor尺寸，分别为 (32, 64, 128, 256, 512)，对应rpn_feature_maps的[P2, P3, P4, P5, P6]，分辨率依次为[256,128,64,32,16]，也就是说底层高分辨率特征去检测较小的目标，顶层低分辨率特征图用于去检测较大的目标。
  - **RPN_ANCHOR_RATIOS**是锚框的长宽比，对应每一种尺寸的锚框取[0.5, 1, 2]，**3种长宽比**
  - **BACKBONE_STRIDES** 是特征图的降采样倍数，取[4, 8, 16, 32, 64]
  - **BACKBONE_SHAPE**是特征图分辨率，为[16,32,64,128,256]

- **generate_anchors**是具体的为每一特征层生成anchor的函数，**generate_pyramid_anchors**用于拼接不同scale的anchor，最终得到anchors的shape为[anchor_count, (y1, x1, y2, x2)]，此时计算的anchor_count = (256×256 + 128×128 + 64×*64 + 32×32 + 16×16)*3 = 261888。数量如此多的锚框不可能全部用于预测，所以有了后续的proposallayer进行筛选。

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\anchors.jpg" style="zoom:67%;" />



#### 3.2 RPN 网络

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\rpn_1.jpeg" style="zoom:67%;" />

FPN输出的特征映射张量[P2, P3, P4, P5, P6]将用于RPN网络，P[?]代表的张量形状各不相同，因此需要分开处理，RPN网络的结构如下图所示，以P2为例：

* P2的大小为256×256@256，经过一次（512，3，3）的卷积操作，形状变为[256, 256, 512]
* 然后分成两路：一路用于目标概率的判断，另一路用于目标框坐标的推断
  * Objectness：
    * 利用1×1的全卷积进行特征维度的压缩。由于每个目标框的anchors数目为3，并且需要分别给出前景和背景的概率值，因此，特征维度需要压缩为2×anchors_per_location个，也就是6个，输出形状为[256, 256, 6]
    * 将[256, 256, 6]的输出重塑为[ 256×256×3,2]，其中第一维度代表anchors的总个数，第二维度代表前景和背景的概率值，最后，通过softmax函数，将概率值转换为类别（One-hot编码）
  * RPN bbox predictions：
    * 利用1×1的全卷积进行特征维度的压缩，由于每个目标框的anchors数目为3，一个box的定位需要4个坐标值（x1，y1，x2，y2），所以一个像素点所需要的的向量长度为3×4=12，然后，将anchors数目在空间维度进行融合，得到256×256×3个anchor，输出的张量形状为[256×256×3, 4]
* 其他特征张量的处理方式上同，作者在这里采用了循环处理的方式，并且共享了RPN的权重。



**注意：以上均是以一张图片的处理过程为例，在实际的操作中，还有batchsize一个维度加在张量的最前面。**

**最终输出形式：**

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\rpn_2.jpg" style="zoom:67%;" />

#### 3.3 Proposal Layer

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\proposallayer.jpeg" style="zoom:67%;" />

Proposal Layer的输入来自于前两部分：Anchors生成和RPN输出。在Proposal Layer中，并没有需要进行训练的参数，只是利用算法进行anchors的选择与修正工作。

- proposallayer需要3个输入参数依次为**[rpn_class, rpn_bbox, anchors]**，自定义层内部表示为**[scores, deltas, anchors]**，其中rpn_class和rpn_bbox为2.2预测分类和回归框偏移，2.1生成的anchors，各有261888个。在训练mrcnn时，一张图片不可能用这么多rois，因此要进一步筛选。
- **Pre-NMS top k：**用于根据rpn_class概率(也可以理解为rois得分)选取至多前6000个rois的索引，然后再根据索引选择出相应的属于top-6000的[scores, deltas, anchors]。tf.gather用于根据y索引选择x值。

* **Box decoding：**用于根据deltas对anchors进行精修
* **removal of invalid boxes**用于将超出图片范围的anchors进行剔除，这里由于回归框是归一化在[0,1]区间内，所以通过clip进行限定。
* **nms**执行非极大值抑制，根据IoU阈值选择出2000个rois，如果选择的rois不足2000，则用0进行pad填充。
* 最终返回的proposals赋值给rpn_rois，作为rpn网络提供的建议区，注入后续FPN heads进行分类、目标框和像素分割的检测。（图中Concatenate boxes from all feature maps of FPN应该是pytorch版本的实现方法，在本结构中，在rpn网络输出的时候就已经将各特征层融合在一起了。）

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\proposallayer1.jpg" style="zoom:67%;" />



### 4 **Network Heads** —— Part3

利用rpn_feature_map和anchors提取出rois，rois和maskrcnn_feature_map一起输入到网络的第三部分：Network Heads。

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\Mask R-CNN framework.png" style="zoom:38%;" />

#### 4.1 RoIAlign layer

RoIAlign的作用是根据rois的坐标，在feature_map中将对应的区域框选出来，然后利用插值法则，生成固定尺寸的张量输入到分类网络和分割网络。这里有两个概念需要理清楚。

* RoIAlign的概念

  离散数据和连续数据不一样，由于rois中的坐标值是离散的，而feature_map中的值是离散的，所以在进行区域截取的时候必然会发生区域的错位现象，为了保证区域的对齐，作者在这里采用了一种叫做RoIAlign的算法，利用二次插值，计算出rois坐标位置的插值，然后进行点的计算与下采样，这样得到的fixed_size_feature_map与rois标注的区域差别最小。

  <img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\roialign.jpeg" style="zoom:67%;" />

  <center> 4.1.1 feature_map和rois</center>

  <img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\roialign_old.jpeg" style="zoom: 67%;" />

  <center> 4.1.2 fast_rcnn采用的区域选择方式</center>

  <img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\rois_new.jpeg" style="zoom:67%;" />

  <center> 4.1.3 mask_rcnn采用的区域对齐方式</center>

* feature_map的选择

  从2.3节中有提到，mrcnn_feature_maps是[P2, P3, P4, P5]的组合向量，每一个输出的特征图大小各不相同，每一个rois需要对应其中的1张特征图进行截取，所以对于特征图的选择需要有一定的规则。这里作者定义了一个系数Pk，其定义为：

$$
k = \lfloor k_0+log_2(\sqrt{wh}/224) \rfloor
$$

其中，$ k_0 $ 代表的是基准值，设置为5，代表P5层的输出，$w, h$ 代表的是roi的长度和宽度， 224是ImageNet的标准输入大小。例如，若$w=112, h=112$， 那么 $k$  的取值为4，rois需要截取P4输出的特征值

#### 4.2 结构

<img src="C:\Users\15214\OneDrive - stu.hit.edu.cn\手册\MaskRCNN manual\imgs\原理解析\network_heads.jpg" style="zoom:80%;" />

整体**FPN heads**分为两个分支，一是用于**分类**和**回归目标框偏移**的**fpn_classifier_graph**，一是用于**像素分割**的**build_fpn_mask_graph**，两者都是将rois在对应mrcnn_feature_maps特征层进行roialign特征提取，然后再经过各自的卷积操作预测最终结果。

**[PyramidROIAlign](https://github.com/matterport/Mask_RCNN/blob/41e7c596ebb83b05a4154bb0ac7a28e0b9afd017/mrcnn/model.py#L344)**层不展开讨论，可认为将pool_size划分的7*7区域(对于mask_pool_size则为14*14)取若干采样点后，进行双线性插值得到f(x,y)，这个版本的代码中取采样点为1。

**[keras.layers.TimeDistributed](https://keras.io/zh/layers/wrappers/)**作为封装器可将一个层应用于输入的每个时间片，要求输入的第一个维度为时间步。该封装器在搭建需要独立连接结构时用到，例如mask rcnn heads结构，进行类别判断、box框回归和mask划分时，需要对num_rois个感兴趣区域ROIs进行分类回归分割处理，每一个区域的处理是相对独立的，此时等价于时间步为num_rois，

##### 4.2.1 **fpn_classifier_graph代码(分类回归分支)**

首先利用PyramidROIAlign提取rois区域的特征，再利用TimeDistributed封装器针对num_rois依次进行7*7->1*1卷积操作，再分出两个次分支，分别用于预测分类和回归框。解释如下：

* PyramidROIAlign用于提取rois区域特征，输出维度为[batch, num_boxes, 7,7,256]
* TimeDistributed封装器针对num_rois依次进行7*7->1*1卷积操作，维度变化为[batch, num_boxes, 1,1,1024]，此时num_rois相当于时间步独立操作。
* 最后输出内容为[mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]，其中mrcnn_probs是预测概率，mrcnn_bbox是预测目标框偏移量

##### 4.2.2 **build_fpn_mask_graph代码(像素分割分支)**

首先利用PyramidROIAlign提取rois区域的特征，再利用TimeDistributed封装器针对num_rois依次进行3*3->3*3->3*3->3*3卷积操作，再经过2*2的转置卷积操作，得到像素分割结果。解释如下：

- PyramidROIAlign用于提取rois区域特征，输出维度为[batch, num_boxes, 14,14,256]
- 4层常规卷积层整合特征
- 最终输出Masks做为分割结果，维度为[batch, num_rois, 28, 28, 80]，这里为每一类实例都提供一个channel，原论文的观点是"*避免了不同实例间的种间竞争"。*