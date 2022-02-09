from torch import nn

class Discriminator(nn.Module):

    def __init__(self, modality):
        super(Discriminator, self).__init__()
        self.modality = modality
        if modality == 'sequences':
            # takes the shared represenataion (output of Encoder) as input
            # first try using 2 fc layers since just numerical data
            hidden_size = 128
            output_size1 = 10
            # assuming only 2 datasets to be merged, 
            # this number indicates whether the shared representation comes from dataset0 or dataset1
            output_size2 = 1 
            self.fc1 = nn.Linear(hidden_size, output_size1)
            self.fc2 = nn.Linear(output_size1, output_size2)
            self.sigmoid = nn.Sigmoid()
        elif modality == 'images':
            # for scanpaths generated from eye-tracking sequences
            # first try following the advice from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            # assume input is (nc) x 64 x 64 (shared reprensentation is also an image)
            nc = 3 # number of color channels
            ndf = 64 # size of feature maps in discriminator
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif modality == 'text':
            pass

    def forward(self, input):
        if self.modality == 'sequences':
            output = self.fc1(input)
            output = self.fc2(output)
            output = self.sigmoid(output)
            return output
        elif self.modality == 'images':
            return self.main(input)
        elif self.modality == 'text':
            pass
