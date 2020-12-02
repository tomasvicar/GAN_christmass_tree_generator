import torch
import nets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave,imread


from dataloader import DataLoader


if __name__ == "__main__":
    
    data_path = '../data'
    workers = 4
    batch_size = 128
    lr = 0.0005
    num_epochs = 300
    latent_size = 100
    c = 0.01
    D_iter =5
    
    device = torch.device("cuda:0")
    
    
    
    
    
    loader = DataLoader(data_path)
    loader = torch.utils.data.DataLoader(loader, batch_size=batch_size,
                                             shuffle=True, num_workers=workers,drop_last=True)
    
    
    G = nets.Generator(latent_size)
    D = nets.Discriminator()
    
    G = G.to(device)
    D = D.to(device)
    
    criterion = nn.BCELoss()
    
    
    
    # optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    
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
            loss_D =-torch.mean( Dx - DGz)
            
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
                
                img = Gz[0,:,:,:].cpu().detach().numpy()*0.5 + 0.5
                
                img = np.transpose(img,(1, 2, 0))
                
                plt.imshow(img)
                plt.show()
            
                imsave('../tmp/' + str(iters).zfill(7) + '.png',img)
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    


