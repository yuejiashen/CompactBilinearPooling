# Compact Bilinear Pooling for Torch 

This code is revised from [@guopei's](https://github.com/guopei) [Compact Bilinear Pooling for Torch](https://github.com/guopei/CompactBiPooling). The main changes include:

1. In the old version, the ouputs, i.e. bilinear features is spatially sum pooled. Now, you can set the **sum_pool = false** if you need some spatial resolution in the output, such as keypoint detection.
2. the new version of package **tcbp** is **1.0-2**(Torch Compact Bilinear Pooling ) to avoid confusion.
3. new tests.

The compact bilinear pooling layer is proposed by Yang Gao etc. in the paper [Compact Bilinear Pooling](https://arxiv.org/abs/1511.06062). This method reduces the spatial complexity of [Bilinear Pooling](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) so that it's feasible for real world training and provides a possible direction to interpret the huge success in fine grained recognition using Bilibear Pooling. We refer you to [caffe implementation page](https://github.com/gy20073/compact_bilinear_pooling) for further information.

## Installation

```
git clone https://github.com/yuejiashen/CompactBilinearPooling
cd cbp
luarocks make rocks/tcbp-1.0-2.rockspec
```

## Test
```
th test.lua
```
Read test.lua for usage.

## Usage
**Example1**
```
require 'cutorch'
require 'tcbp'
net = nn.ComBiPooling(1024, false):cuda()  -- set sum pool to false, spatial resoultion will be reserved
input1 = torch.rand(10, 300, 7, 7):cuda()  -- batch size = 10, input dim = 300, height = 7, width = 7
input2 = torch.rand(10, 300, 7, 7):cuda()
input = {input1, input2}
output = net:forward(input)
print(output:size())

   10
 1024
    7
    7
[torch.LongStorage of size 4]
```
**Example2**
```
require 'cutorch'
require 'tcbp'
net = nn.ComBiPooling(1024):cuda()  -- by default, sum pool is set to true, spatial resolution will be sumed
input1 = torch.rand(10, 300, 7, 7):cuda()  -- batch size = 10, input dim = 300, height = 7, width = 7
input2 = torch.rand(10, 300, 7, 7):cuda()
input = {input1, input2}
output = net:forward(input)
print(output:size())

   10
 1024
[torch.LongStorage of size 2]
```
## References
1. [spectral-lib](https://github.com/mbhenaff/spectral-lib)
2. [cbp](https://github.com/jnhwkim/cbp)
3. [compact_bilinear_pooling](https://github.com/gy20073/compact_bilinear_pooling)
4. [tensorflow_compact_bilinear_pooling](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)
5. [tcbp](https://github.com/guopei/CompactBiPooling)
