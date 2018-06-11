import numpy as np
import torch
import Interp
import numbers

class SynthData(object):
    def __init__(self, size, n=1, reset_after=250):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))

        # coordinate grid
        x,y=np.meshgrid(np.linspace(-1,1,size[1]+2),np.linspace(1,-1,size[0]+2))
        self.x=torch.from_numpy(x).float()
        self.y=torch.from_numpy(y).float()
        # height field
        self.u=[torch.zeros(size[0]+2,size[1]+2) for _ in range(n)]
        # velocity field
        self.v=[torch.zeros(size[0]+2,size[1]+2) for _ in range(n)]

        self.n = n # number of simulations (for independent batches)
        self.reset_after=reset_after
        self.reset()

    def reset(self):
        for u in self.u+self.v:
            u.fill_(0)
        self.ct=0
        self.random_config()
        #for i in range(self.n):
        #    self.step(i, 50)

    def random_config(self):
        # initialize random distribution for perturb to sample from
        self.damp        = np.random.rand(self.n)*.005+.994
        self.motion_blur = np.random.randint(7,15, size=self.n)
        self.window      = np.hstack(( -np.ones((self.n,1)), np.ones((self.n,1)), -np.ones((self.n,1)), np.ones((self.n,1)) ))
        self.size        = np.hstack(( .4-np.random.rand(self.n,1)*.25, .4+np.random.rand(self.n,1)*.25 ))
        self.ecin        = np.hstack(( .7-np.random.rand(self.n,1)*.25, .7+np.random.rand(self.n,1)*.25 ))
        self.strength    = np.hstack(( 10-np.random.rand(self.n,1)*4, 10+np.random.rand(self.n,1)*4 ))
        theta0           = np.random.rand(self.n,1)*180*np.pi/180
        thetad           = np.random.rand(self.n,1)*30*np.pi/180
        self.theta       = np.hstack(( theta0-thetad , theta0+thetad ))
        self.prob        = np.random.rand(self.n)*.1+.1
        

    def perturb(self, i=0):
        # perturb the surface
        if np.random.rand()>self.prob[i]:
            return

        size = np.random.rand()**5*(self.size[i,1]-self.size[i,0])+self.size[i,0]
        strength = np.random.rand()*(self.strength[i,1]-self.strength[i,0])+self.strength[i,0]
        xc=np.random.rand()*(self.window[i,1]-self.window[i,0])+self.window[i,0]
        yc=np.random.rand()*(self.window[i,3]-self.window[i,2])+self.window[i,2]
        ecin=np.random.rand()*(self.ecin[i,1]-self.ecin[i,0])+self.ecin[i,0]
        theta=np.random.rand()*(self.theta[i,1]-self.theta[i,0])+self.theta[i,0]

        x = (self.x-xc)*np.cos(theta) + (self.y-yc)*np.sin(theta)
        y = -(self.x-xc)*np.sin(theta) + (self.y-yc)*np.cos(theta)
        r2=(x-xc)**2/size**2+(y-yc)**2/(1-ecin)/size**2
        self.u[i] += np.random.choice([-1,1])*torch.exp(-r2)*strength*25

    def step(self, i=0, steps=1, lock=False):
        # step the simulation
        u=self.u[i]
        v=self.v[i]
        for _ in range(steps):
            if not lock:
                self.perturb(i)
            f1 = (u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] - 4*u[1:-1,1:-1])/4
            f2 = (u[:-2,:-2] + u[2:,:-2] + u[:-2,2:] + u[2:,2:] - 4*u[1:-1,1:-1])/2/4
            v[1:-1,1:-1] = v[1:-1,1:-1] + (f1+f2)
            v*=self.damp[i]
            u+=v
            u[0,:]=u[1,:]*1
            u[:,0]=u[:,1]*1
            u[-1,:]=u[-2,:]*1
            u[:,-1]=u[:,-2]*1



    def __call__(self, img):
        # warp an image
        if self.ct >= self.reset_after*self.n:
            self.reset()
        i = self.ct % self.n
        img1 = torch.zeros_like(img)
        self.step(i, 10)
        for j in range(self.motion_blur[i]):
            if j>0:
                self.step(i, 2, lock=True)
            ux=(self.u[i][1:-1,2:]-self.u[i][1:-1,:-2])/2
            uy=(self.u[i][2:,1:-1]-self.u[i][:-2,1:-1])/2
            img1+=Interp.warp(img.unsqueeze(0),ux.unsqueeze(0),uy.unsqueeze(0)).squeeze(0)

        img1 = img1/self.motion_blur[i]
        self.ct+=1
        return [img1, img]


if __name__ == '__main__':
    # test visualizations

    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()

    x = SynthData(256, n=1)
    """
    for i in range(100):
        x.step(steps=3)
        plt.imshow(x.u[0].numpy())
        plt.title('%d %f %f %f'%(i,x.u[0].min(), x.u[0].mean(), x.u[0].max()))
        plt.draw()
        plt.pause(.1)


    """
    from datasets import ImageFolder
    import torchvision.transforms as transforms

    img_dir='/mnt/Data1/ImageNet/val'
    data = ImageFolder(img_dir, transform=
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]))
    img = data[10][0]

    for i in range(100):
        img1 = x(img)[0]
        plt.imshow(img1.permute(1, 2, 0))
        plt.draw()
        plt.pause(.1)


