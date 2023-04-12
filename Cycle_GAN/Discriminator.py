''' this is the Discriminator_Model from scratch implementaion of Cycle_GAN
by Levi_Ack at 2023/1/14

a more detailed look of the mechanism behind the Cycle_GAN can be found in:
https://www.youtube.com/watch?v=lhs78if-E7E&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=10

'''

__author__ ='Levi_Ack'

import torch
import torch.nn as nn

'''basically each of this dis_block will decrease the img_size(w/h) ->(1/2)img_size(w/h) '''
class Dis_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True),
        )

    def forward(self,x):

        return self.conv(x)
# here the channels of the feature_map will go though 64->128->256->512 as the original paper
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator,self).__init__()
        self.initial_layer=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, ##here the input_channels is the channels of the origin pic ,that's 3
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(0.2,inplace=True),
        )

        layers=[]  # used as a container to the down_sampling blocks
        new_in_channels=features[0]   #the initial_layer will transfer the channels of the tensor to features[0]
        
        for feature in features[1:]:  # here the first one is used in initial_layer
            layers.append(
                Dis_Block(
                    in_channels=new_in_channels,
                    out_channels=feature,
                    stride=1 if feature==features[-1] else 2 ## in the last layer we use a stride of 1 else 2 to decrease the size of features
                )
            )
            new_in_channels=feature #update the in_channels_value of the next block
        
        layers.append(
            nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode='reflect',
            )
        )
        self.Dis_model=nn.Sequential(*layers)

    def forward(self,x):
        x=self.initial_layer(x)
        x=self.Dis_model(x)
        final_prob_of_each_patch=torch.sigmoid(x)  ##here we implement the sigmoid() function to get the real/fake--prob of each patchs
        # we also can use nn.Sigmoid() too

        return final_prob_of_each_patch

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

    # print(preds)


if __name__ == "__main__":
    test()