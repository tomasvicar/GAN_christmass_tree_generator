import torch
import nets_64,nets_128
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave,imread
import torchvision.utils as vutils

from dataloader import DataLoader


if __name__ == "__main__":
    
    data_path = '../data_64'
    workers = 6
    batch_size = 64
    lr = 0.00005
    num_epochs = 300
    latent_size = 100
    c = 0.01
    lam = 10 
    D_iter = 5
    
    device = torch.device("cuda:0")
    
    
    
    
    
    loader = DataLoader(data_path)
    loader = torch.utils.data.DataLoader(loader, batch_size=batch_size,
                                             shuffle=True, num_workers=workers,drop_last=True)
    
    
    G = nets_64.Generator(latent_size)
    D = nets_64.Discriminator()
    
    G = G.to(device)
    D = D.to(device)
        
    
    
    # optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    # optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    
    optimizerD = optim.RMSprop(D.parameters(), lr=lr)
    optimizerG = optim.RMSprop(G.parameters(), lr=lr)
    
    G_losses = []
    D_losses = []
    
    iters =-1
    for epoch in range(num_epochs):
        
        for i,data in enumerate(loader):
            iters = iters +1
            
            
             # minimize -(log(D(x)) + log(1 - D(G(z)))
            
            x = data.to(device)
    
            z = torch.randn(batch_size, latent_size, 1, 1, device=device)
            
            Gz = G(z).detach()
               
            Dx = D(x)
            
            DGz = D(Gz)
            
            
            # loss_D =-torch.mean( torch.log(Dx) + torch.log(1 - DGz))
            
            # eps = torch.rand(batch_size, 1,device=device)
            # s=Dx.size()
            # eps = eps.expand(batch_size, int(Dx.nelement()/batch_size)).contiguous().view(s[0],s[1],s[2],s[3])
            # x_hat = eps * x + ((1 - eps) * Gz)
            # x_hat = x_hat.detach()
            # x_hat.requires_grad = True
            # Dx_hat = D(x_hat)
            # gradDx_hat = torch.autograd.grad(outputs=Dx_hat, inputs=x_hat,grad_outputs=torch.ones(Dx_hat.size(),device=device),create_graph=True, retain_graph=True, only_inputs=True)[0]
            # gradDx_hat = gradDx_hat.view(gradDx_hat.size(0), -1)
            # gradient_penalty = ((gradDx_hat.norm(2, dim=1) - 1) ** 2)

            
            # loss_D =torch.mean(DGz) - torch.mean(Dx) + lam*torch.mean(gradient_penalty )
            
            loss_D =torch.mean(DGz) - torch.mean(Dx)
            
            D.zero_grad()
            loss_D.backward()
            optimizerD.step()
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)
            
        
            D_losses.append(loss_D.cpu().detach().numpy())
            
            
            
            if (iters%D_iter) ==0:
                # minimize  -log(D(G(z)))
                
                z = torch.randn(batch_size, latent_size, 1, 1, device=device)
                
                Gz = G(z)
                
                DGz = D(Gz)
                
                # loss_G = -torch.mean(torch.log(DGz))
                loss_G = -torch.mean(DGz)
                
                G.zero_grad()
                loss_G.backward()
                optimizerG.step()
            
            
                G_losses.append(loss_G.cpu().detach().numpy())
            
            
            
            if (iters % 100 == 0):
                plt.plot(D_losses)
                plt.title('D')
                plt.show()
                plt.plot(G_losses)
                plt.title('G')
                plt.show()
                
                # img = Gz[0,:,:,:].cpu().detach().numpy()*0.5 + 0.5
                # img = np.transpose(img,(1, 2, 0))
                # plt.imshow(img)
                # plt.show()
                
                
                img = np.transpose(vutils.make_grid(Gz.cpu().detach()[:32],padding=2, normalize=True).numpy(),(1,2,0))
                
                plt.figure(figsize=(15,15))
                plt.imshow(img)
                plt.show()
                
                imsave('../tmp/' + str(iters).zfill(7) + '.png',img)
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    



