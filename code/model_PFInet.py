import torch
import torch.nn as nn

def SAI2MacPI(x, angRes):
    b,c,h,w = x.shape
    b = b//angRes//angRes
    SAI = x.contiguous().view(b,angRes,angRes,c,h,w)

    SAI = SAI.permute(0,3,4,1,5,2)   
    SAI = SAI.reshape(b,c,angRes*h,angRes*w)
   
    return SAI

def MacPI2SAI(x, angRes):
    b,c,H,W = x.shape
    h = int(H//angRes)
    w = int(W//angRes)
    
    MPI = x.reshape(b,c,h,angRes,w,angRes)
    SAI = MPI.permute(0,1,3,5,2,4)
    SAI = SAI.contiguous().view(b,angRes,angRes,c,h,w)
    return SAI

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Attention(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out
    
class DRBN_BU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DRBN_BU, self).__init__()

        G0 = 64
        kSize = 3
        self.D = 6 
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(in_channels, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()
        
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(G0*2, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, out_channels, kSize, padding=(kSize-1)//2, stride=1)
            ])

        
        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0*2, kSize, padding=(kSize-1)//2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0*2, G0, kSize+1, stride=2, padding=1) 

        self.Relu = nn.LeakyReLU(0.1, inplace=True)
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        self.gate1 = Attention(2*G0)
        self.gate2 = Attention(G0)
        self.gate3 = Attention(G0)
        
    def forward(self, x):
        
        f_first = self.Relu(self.SFENet1(x))
        f_s1  = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1)) 
        f_s4 = self.Down2(self.RDBs[1](f_s2))
   
        
        f_s4 = self.gate1(f_s4) + self.RDBs[3](self.RDBs[2](f_s4))
        f_s2 = self.gate2(f_s2) + self.RDBs[4](self.Up2(f_s4))
        f_s1 = self.gate3(f_s1) + self.RDBs[5](self.Up1(f_s2))+f_first
     

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 32
        angRes = 5
        n_blocks = 4
        self.SFE =  nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.AFE = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), bias=False)
        self.EPIConv_h = nn.Conv2d(channels, channels, kernel_size=[1, angRes * angRes], 
                            stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False)
        self.EPIConv_v = nn.Conv2d(channels, channels, kernel_size=[angRes * angRes, 1], 
                            stride=[angRes, 1], padding=[angRes * (angRes - 1)//2, 0], bias=False)
  
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.block = CascadeInterBlock(angRes, n_blocks, channels)
        
        self.a2s = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.h2s = nn.Sequential(
            nn.Conv2d(channels, angRes * channels, kernel_size=1, stride=1, padding=0, bias=False),
            PixelShuffle1D(angRes),
        )
        self.out_conv = nn.Conv2d(4*channels, 3, kernel_size=1, bias=False)
        
        self.SAI_stack0 = DRBN_BU(3*angRes*angRes, channels)
        self.SAI_stack1 = DRBN_BU(3*9, channels)
    
    def prepare_data1(self, lf):
        N, an2, c, h, w = lf.shape
        an = 5
        
        focal_stack = torch.zeros((N, an, an, c * 9, h, w))
        x = lf.reshape(N, an * an, c, h * w)

        x = x.reshape(N, an * an, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, h * w, c, an, an)
        # x = lf.view(N, an, an, c, h, w)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)

        x = x.reshape(N, h * w, c, (an + 2) * (an + 2))
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, (an + 2) * (an + 2), c, h, w)
        x = x.reshape(N, (an + 2), (an + 2), c, h, w)

        x6 = x[:, :-2, :-2].reshape(N, an2, c, h, w)
        x2 = x[:, :-2, 1:-1].reshape(N, an2, c, h, w)
        x8 = x[:, :-2, 2:].reshape(N, an2, c, h, w)
        x4 = x[:, 1:-1, :-2].reshape(N, an2, c, h, w)

        x3 = x[:, 1:-1, 2:].reshape(N, an2, c, h, w)
        x7 = x[:, 2:, :-2].reshape(N, an2, c, h, w)
        x1 = x[:, 2:, 1:-1].reshape(N, an2, c, h, w)
        x5 = x[:, 2:, 2:].reshape(N, an2, c, h, w)
        focal_stack = torch.cat([x6, x2, x8, x4, lf, x3, x7, x1, x5], dim=2)

        return focal_stack

    def my_norm(self, x):
        N, an2, c, h, w = x.shape
        lf_avg = torch.mean(x, dim=1, keepdim=False)  # [N, c, h, w]
        gray = 0.2989 * lf_avg[:, 0, :, :] + 0.5870 * lf_avg[:, 1, :, :] + 0.1140 * lf_avg[:, 2, :, :]  # [N, h, w]
        temp = (1 - gray) * gray
        ratio = (h * w) / (2 * torch.sum(temp.reshape(N, -1), dim=1))
        return ratio
       
       
    def forward(self, x):
        b,u,v,c,h,w = x.shape
        x_res = x
        x = x.reshape(b,u*v,c,h,w)

        ######### can be choosed #########
        # ratio = self.my_norm(x).reshape(b, 1, 1, 1, 1).expand_as(x)
        # x = x * ratio
        ##################################
        
        x_inter = x.reshape(b,u*v*c,h,w)
        x_intra = self.prepare_data1(x) # b n c*9 h w
        x_intra = x_intra.reshape(b*u*v,c*9,h,w)
        
        x_inter = self.SAI_stack0(x_inter)
        x_intra = self.SAI_stack1(x_intra)
        
        # print(x_inter.shape, x_intra.shape)
        x_en = self.fusion(torch.cat([x_inter.repeat(u*v,1,1,1),x_intra], dim=1))
        x_MPI = SAI2MacPI(x_en, u)
        
        xa = self.AFE(x_MPI)
        xs = self.SFE(x_MPI)
        xh = self.EPIConv_h(x_MPI)
        xv = self.EPIConv_v(x_MPI)
        
        xa, xs, xh, xv = self.block(xa, xs, xh, xv)
        
        xas = self.a2s(xa)
        xhs = self.h2s(xh)
        xvs = self.h2s(xv.permute(0,1,3,2)).permute(0,1,3,2)
        out = torch.cat([xas,xs,xhs,xvs], 1)
        out = self.out_conv(out)
        out = MacPI2SAI(out, u) + x_res
        return out.reshape(b,u*v*c,h,w)
       

class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y
    
    
class A_HV_Fuse(nn.Module):
    def __init__(self, channels, angRes):
        super(A_HV_Fuse, self).__init__()

        self.EPIConv_h = nn.Conv2d(channels, channels, kernel_size=[1, angRes * angRes], 
                            stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False)
        self.EPIConv_v = nn.Conv2d(channels, channels, kernel_size=[angRes * angRes, 1], 
                            stride=[angRes, 1], padding=[angRes * (angRes - 1)//2, 0], bias=False)
        
        
        
        self.EUP = nn.Sequential(
            nn.Conv2d(channels, angRes * channels, kernel_size=1, stride=1, padding=0, bias=False),
            PixelShuffle1D(angRes),
        )
        
        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, xa, x_h, x_v):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.EPIConv_v(x_h))
        buffer_ang3 = self.ReLU(self.EPIConv_h(x_v))
       
        buffer_eh = self.EUP(xa)
        buffer_ev = self.EUP(xa.permute(0,1,3,2)).permute(0,1,3,2)
        
        buffer_a = torch.cat((buffer_ang1, buffer_ang2, buffer_ang3), 1)
        buffer_h = torch.cat((buffer_ev, x_h), 1)
        buffer_v = torch.cat((buffer_eh, x_v), 1)
      
        out_a = self.ReLU(self.conv1(buffer_a)) + xa
        out_h = self.ReLU(self.conv2(buffer_h)) + x_h
        out_v = self.ReLU(self.conv2(buffer_v)) + x_v
        
        return out_a, out_h, out_v

class S_HV_Fuse(nn.Module):
    def __init__(self, channels, angRes):
        super().__init__()
        self.EUP = nn.Sequential(
            nn.Conv2d(channels, angRes * channels, kernel_size=1, stride=1, padding=0, bias=False),
            PixelShuffle1D(angRes),
        )
        
        self.EPIConv_h = nn.Conv2d(channels, channels, kernel_size=[1, angRes * angRes], 
                            stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False)
        self.EPIConv_v = nn.Conv2d(channels, channels, kernel_size=[angRes * angRes, 1], 
                            stride=[angRes, 1], padding=[angRes * (angRes - 1)//2, 0], bias=False)
   
        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, xs, x_h, x_v):
        buffer_spa1 = xs
        buffer_spa2 = self.EUP(x_h)
        buffer_spa3 = self.EUP(x_v.permute(0,1,3,2)).permute(0,1,3,2)
        
        buffer_eh = self.EPIConv_h(xs)
        buffer_ev = self.EPIConv_v(xs)
        
        buffer_a = torch.cat((buffer_spa1, buffer_spa2, buffer_spa3), 1)
        buffer_h = torch.cat((buffer_eh, x_h), 1)
        buffer_v = torch.cat((buffer_ev, x_v), 1)
      
        out_s = self.ReLU(self.conv1(buffer_a)) + xs
        out_h = self.ReLU(self.conv2(buffer_h)) + x_h
        out_v = self.ReLU(self.conv2(buffer_v)) + x_v
        
        return out_s, out_h, out_v
 
class A_S_HV_Fuse(nn.Module):
    def __init__(self, channels, angRes):
        super().__init__()   
        self.AFE = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        
        self.AUP = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv1 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, xa, xs, x_h, x_v):
       
        
        fea_xa = torch.cat([xa, self.AFE(xs)], 1)
        fea_xs = torch.cat([xs, self.AUP(xa)], 1)
        
        xa = self.ReLU(self.conv1(fea_xa)) + xa
        xs = self.ReLU(self.conv1(fea_xs)) + xs
        
        map_xh = self.conv_h(x_v).permute(0,1,3,2)
        map_xv = self.conv_v(x_h).permute(0,1,3,2)
        xh = torch.cat([x_h, map_xh], 1)
        xv = torch.cat([x_v, map_xv], 1)
        
        x_h = self.ReLU(self.conv2(xh)) + x_h
        x_v = self.ReLU(self.conv2(xv)) + x_v
        
        return xa, xs, x_h, x_v

class make_chains(nn.Module):
    def __init__(self, channels, angRes) :
        super().__init__()
        self.a_hv = A_HV_Fuse(channels, angRes)
        self.s_hv = S_HV_Fuse(channels, angRes)
        self.a_s_hv = A_S_HV_Fuse(channels, angRes)
    
    def forward(self, xa, xs, xh, xv):
        xa, xh, xv = self.a_hv(xa, xh, xv)
        xs, xh, xv = self.s_hv(xs, xh, xv)
        xa, xs, xh, xv = self.a_s_hv(xa, xs, xh, xv)
        return xa, xs, xh, xv

class CascadeInterBlock(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(CascadeInterBlock, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(make_chains(channels, angRes))
        self.body = nn.Sequential(*body)
        
    def forward(self, buffer_a, buffer_s, buffer_h, buffer_v):
        out_a = []
        out_s = []
        out_h = []
        out_v = []
        for i in range(self.n_blocks):
            buffer_a, buffer_s, buffer_h, buffer_v = self.body[i](buffer_a, buffer_s, buffer_h, buffer_v)
            if i == 0:
                out_a = buffer_a
                out_s = buffer_s
                out_h = buffer_h
                out_v = buffer_v
            else:
                out_a = out_a + buffer_a
                out_s = out_s + buffer_s
                out_h = out_h + buffer_h
                out_v = out_v + buffer_v
        return out_a, out_s, out_h, out_v
        
if __name__ == "__main__":
    net = Net()
    from thop import profile
    input = torch.randn(1,5,5,3,64,64)
    out = net(input)
    print(out.shape)
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))