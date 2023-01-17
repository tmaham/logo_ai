from torch import nn
import pdb 
import torch 

dcgan = True

if dcgan:
    class Discriminator(nn.Module):
        def __init__(self, input_dims, hidden_dims):
            super(Discriminator, self).__init__()
            self.ngpu = 0
            self.mode = 12

            if self.mode == 1:
                nc = 5
                o = 1
            else:
                nc = 4
                o = 26

            ndf = 64

            self.relu = nn.LeakyReLU(0.2, inplace=True)

            self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
            # state size. (ndf) x 32 x 32
            self.layer2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(ndf * 2)
            # state size. (ndf*2) x 16 x 16
            self.layer3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(ndf * 4)
            # state size. (ndf*4) x 8 x 8
            self.layer4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            self.bn4 = nn.BatchNorm2d(ndf * 8)
            # state size. (ndf*8) x 4 x 4
            self.layer5 = nn.Conv2d(ndf * 8, o, 2, 1, 0, bias=False)
            self.sig = nn.Sigmoid()
        

        def forward(self, input, letter):
            if self.mode == 1:
                letter = letter/26
                letter_layer = torch.ones([1,1,32,32])*letter
                input = torch.cat((input, letter_layer.cuda()), dim=1)
            out = self.relu(self.layer1(input))
            out = self.relu(self.bn2(self.layer2(out)))
            out = self.relu(self.bn3(self.layer3(out)))
            out = self.relu(self.bn4(self.layer4(out)))
            out = self.sig(self.layer5(out))
            if self.mode is not 1:
                out = out[0][letter]
            return out
else:
    class Discriminator(nn.Module):

        def __init__(self, input_dims, hidden_dims):
            super(Discriminator, self).__init__()

            self.restored = False
            self.relu = nn.LeakyReLU()
            self.sig = nn.Sigmoid()

            self.layer1 = nn.Linear(4096, 4096)
            self.layer2 = nn.Linear(4096, 4096)

            self.final_layer = nn.Linear(4096, 26)
            

        def forward(self, input, letter):
            input = input.flatten(start_dim=1)
            out = self.relu(self.layer1(input))
            out = self.relu(self.layer2(out))
            last = self.sig(self.final_layer(out))
            return last[0][letter]
            return last

class Adapter(nn.Module):

    def __init__(self):
        super(Adapter, self).__init__()

        self.restored = False
        self.relu = nn.LeakyReLU()

        self.layer1 = nn.Linear(4096, 4096)
        self.layer2 = nn.Linear(4096, 4096)
        

    def forward(self, input):
        input = input.flatten(start_dim=1)
        out = self.relu(self.layer1(input))
        out = self.relu(self.layer2(out))
        return out
