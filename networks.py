import torch
import torch.nn.functional as F
from torch import nn
import itertools
from collections import OrderedDict
import Interp
from torchvision.models import vgg16


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # weights of losses
        self.weights = {'X_L1':args.weight_X_L1, 'Y_L1':args.weight_Y_L1, 'Z_L1':args.weight_Z_L1, 'Y_VGG':args.weight_Y_VGG, 'Z_VGG':args.weight_Z_VGG, 'Z_Adv':args.weight_Z_Adv}

        # networks
        params_G = []

        # color corrector network (used to correct color shift between monitor images and real images in the training set)
        if args.weight_X_L1>0:
            self.cc_net = CCNet()
            if args.freeze_cc_net:
                for param in self.cc_net.parameters():
                    param.requires_grad=False
            else:
                params_G = itertools.chain(params_G, self.cc_net.parameters())
        else:
            self.cc_net = None

        # warp net
        if args.warp_net:
            self.warp_net = I2INet(3, 2, args.warp_net_downsample, False, args.dim, args.n_res, args.norm, 'relu', 'reflect', final_activ='none')
            if args.freeze_warp_net:
                for param in self.warp_net.parameters():
                    param.requires_grad=False
            else:
                params_G = itertools.chain(params_G, self.warp_net.parameters())
        else:
            self.warp_net = None

        # color net
        if args.color_net:
            self.color_net = I2INet(3, 3, args.color_net_downsample, args.color_net_skip, args.dim, args.n_res, args.norm, 'relu', 'reflect', args.denormalize)
            params_G = itertools.chain(params_G, self.color_net.parameters())
        else:
            self.color_net = None

        # optimizer
        self.optimizer_G = torch.optim.Adam(params_G, lr=2e-4, betas=(.5, 0.999))

        # for reconstruction loss
        self.recon_criterion = nn.L1Loss()

        # discriminator for adversarial loss
        if self.weights['Z_Adv']>0:
            self.discriminator = Discriminator(dim=args.dim)
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(.5, 0.999))

        # vgg for perceptual loss
        if self.weights['Y_VGG']>0 or self.weights['Z_VGG']>0:
            self.vgg = vgg_features()


    def forward(self, w, cc=True):
        # color corrector network
        if self.cc_net is not None and cc:
            x = self.cc_net(w)
        else:
            x = w

        # warp net
        if self.warp_net is not None:
            warp = self.warp_net(x)*10
            y = Interp.warp(x,warp[:,0,:,:],warp[:,1,:,:])
        else:
            warp = torch.zeros_like(x[:,:2,:,:])
            y = x

        # color net
        if self.color_net is not None:
            z = self.color_net(y)
        else:
            z=y

        return x, warp, y, z


    def compute_loss_G(self, x, y, z, target):
        losses = OrderedDict()

        # Reconstruction loss
        if self.weights['X_L1']>0:
            losses['X_L1'] = self.recon_criterion(x, target)
        if self.weights['Y_L1']>0:
            losses['Y_L1'] = self.recon_criterion(y, target)
        if self.weights['Z_L1']>0:
            losses['Z_L1'] = self.recon_criterion(z, target)

        # Perceptual loss
        if self.weights['Y_VGG']>0:
            losses['Y_VGG'] = self.recon_criterion(self.vgg(y), self.vgg(target))
        if self.weights['Z_VGG']>0:
            losses['Z_VGG'] = self.recon_criterion(self.vgg(z), self.vgg(target))

        # Adversarial loss
        if self.weights['Z_Adv']>0:
            losses['Z_Adv'] = self.discriminator.calc_gen_loss(z)

        return losses


    def optimize_parameters(self, input, target):
        x, warp, y, z = self.forward(input)
        # update discriminator
        if self.weights['Z_Adv']>0:
            self.optimizer_D.zero_grad()
            loss_d = self.discriminator.calc_dis_loss(z, target)
            loss_d.backward()
            self.optimizer_D.step()
        
        # update generators
        self.optimizer_G.zero_grad()
        losses  = self.compute_loss_G(x, y, z, target)
        loss = sum([losses[key]*self.weights[key] for key in losses.keys()])
        loss.backward()
        self.optimizer_G.step()
        
        if self.weights['Z_Adv']>0:
            losses['Dis'] = loss_d
        return losses


    def print_losses(self, losses):
        return ' '.join(['Loss %s: %.4f'%(key, val.item()) for key,val in losses.items()])



class I2INet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8, norm='in', activ='relu', pad_type='reflect', denormalize=False, final_activ='tanh'):
        super(I2INet, self).__init__()

        self.skip=skip
        self.denormalize = denormalize

        # project to feature space
        self.conv_in = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)

        # downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.down_blocks.append( Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type) )
            dim *= 2

        # residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(n_res):
            self.res_blocks.append( ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type) )

        # upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.up_blocks.append( nn.Sequential(nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)) )
            dim //= 2

        # project to image space
        self.conv_out = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=final_activ, pad_type=pad_type)

        #self.apply(weights_init('kaiming'))
        #self.apply(weights_init('gaussian'))


    def forward(self, x):
        # normalize image and save mean/var if using denormalization
        if self.denormalize:
            x_mean = x.view(x.size(0), x.size(1), -1).mean(2).view(x.size(0), x.size(1), 1, 1)
            x_var = x.view(x.size(0), x.size(1), -1).var(2).view(x.size(0), x.size(1), 1, 1)
            x = (x-x_mean)/x_var

        # project to feature space
        x = self.conv_in(x)

        # downsampling blocks
        xs = []
        for block in self.down_blocks:
            xs += [x]
            x = block(x)

        # residual blocks
        for block in self.res_blocks:
            x = block(x)

        # upsampling blocks
        for block, skip in zip(self.up_blocks, reversed(xs)):
            x = block(x)
            if self.skip:
                x = x + skip

        # project to image space
        x = self.conv_out(x)

        # denormalize if necessary
        if self.denormalize:
            x = x*x_var+x_mean
        return x


class CCNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, layers=5, dim=32, norm='gn', activ='relu', pad_type='reflect', final_activ='tanh'):
        super(CCNet, self).__init__()
        self.model = []
        #self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(input_dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(layers-2):
            self.model += [Conv2dBlock(dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation=final_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x)


class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        # get vgg16 features up to conv 4_3
        self.model = nn.Sequential(*list(vgg16(pretrained=True).features)[:23])
        # will not need to compute gradients
        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x, renormalize=True):
        # change normaliztion form [-1,1] to VGG normalization
        if renormalize:
            x = ((x*.5+.5)-torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, gan_type='lsgan', input_dim=3, dim=64, n_layers=4, norm='bn', activ='lrelu', pad_type='reflect'):
        super(Discriminator, self).__init__()
        self.gan_type = gan_type
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)]
        for i in range(n_layers - 1):
            self.model += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        #self.apply(weights_init('gaussian'))

    def forward(self, input):
        return self.model(input).mean(3).mean(2).squeeze()

    def calc_dis_loss(self, input_fake, input_real):
        input_fake = input_fake.detach()
        input_real = input_real.detach()
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(out0, requires_grad=False).cuda()
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                              F.binary_cross_entropy(F.sigmoid(out1), all1))
        elif self.gan_type == 'wgan':
            loss = out0.mean()-out1.mean()
            # grad penalty
            BatchSize = input_fake.size(0)
            alpha = torch.rand(BatchSize,1,1,1, requires_grad=False).cuda()
            interpolates = (alpha * input_real) + (( 1 - alpha ) * input_fake)
            interpolates.requires_grad=True
            outi = self.forward(interpolates)
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            gradients = torch.autograd.grad(outi, interpolates, grad_outputs=all1, create_graph=True)[0]
            #gradient_penalty = ((gradients.view(BatchSize,-1).norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty = ((gradients.view(BatchSize,-1).norm(1, dim=1) - 1).clamp(0) ** 2).mean()
            loss += 10*gradient_penalty
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        out0 = self.forward(input_fake)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 1)**2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(out0.data, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        elif self.gan_type == 'wgan':
            loss = -out0.mean()
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss



##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transposed=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(norm_dim/8, norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transposed:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x




##################################################################################
# weight initialization
##################################################################################

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)
    return init_fun



