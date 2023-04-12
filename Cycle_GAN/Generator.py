''' this is the Discriminator_Model from scratch implementaion of Cycle_GAN
by Levi_Ack at 2023/1/14

a more detailed look of the mechanism behind the Cycle_GAN can be found in:
https://www.youtube.com/watch?v=lhs78if-E7E&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=10

'''
## this structure is a little bit like U-Net but the discrepancy is that we use residual block instead of bottleneck in the bottom layer
__author__ ='Levi_Ack'

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        down=True, ## decide the mode i.e. we are in down-mode(use con_layer f for dowm_sample) up-mode(use convtranspose_layer for up sample--generate pic)
        use_act=True, 
        **kwargs):
        super(ConvBlock,self).__init__()
        ''' one thing notable here is that we only use padding in the down_sample, else may cause artifact(虚像)'''
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode='reflect',
                **kwargs  ## other arguments(like stride/ker_size) are reserved for future specific defination
            )
            if down  # in down mode
            else  # in up mode
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs ## other arguments(like stride/ker_size) are reserved for future specific defination
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(), # when use act else do nothing!
        )

    def forward(self,x):
        return self.conv(x)
'''
the residual blcok is 3x3(with 1 padding)+3x3(with one padding)
'''
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block=nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1
                ),
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
    def forward(self,x):
        res =x
        output =res+self.block(x)

        return output

'''
the overall structure is a lit bit like the U-Net 
but here we use resblock instead of bottleneck which used in pix2pix paper
the outline of the net is:
Down_sample_layer -> resblock -> Up_sample_layer
'''
class Generator(nn.Module):
    def __init__(
        self, 
        img_channels=3, 
        num_features=64, 
        num_residuals=9 
        ):
        super().__init__()
        ''' the initial_layer only used to expanse the img_channels to features and dosen't touch the img_size'''
        self.initial_layer=nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_blocks=nn.ModuleList([
            ConvBlock(
                in_channels=num_features,
                out_channels=num_features*2,
                kernel_size=3,
                stride=2,
                padding=1),
            ConvBlock(
                in_channels=num_features*2,
                out_channels=num_features*4,
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.res_blocks=nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks=nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features*4,
                    out_channels=num_features*2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1),

                    ConvBlock(
                    in_channels=num_features*2,
                    out_channels=num_features,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1)
            ]
        )
        ''' the last up_layer is used to generate the pic transfer the channels to RGB channels'''
        self.final_up=nn.Conv2d(
            in_channels=num_features,
            out_channels=img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode='reflect',
        )
    
    def forward(self,x):
        x=self.initial_layer(x)

        for down_layer in self.down_blocks:
            x=down_layer(x)
        
        x=self.res_blocks(x)

        for up_layer in self.up_blocks:
            x=up_layer(x)

        return torch.tanh(self.final_up(x))  #tanh() was used to scale the pixel value to [-1,1] so to generate the pic, we can also use nn.Tanh() here

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()

