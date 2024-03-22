import torch
from torch import nn
import torch.nn.functional as F
import unet3d


# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.base_out = self.base_model(x)
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2( F.relu(self.linear_out))
        return self.sigmoid(final_out)

# prepare the 3D model
class TargetNetV2(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(TargetNetV2, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        if t>0:
           x1 = x[0]
           x2 = x[1]
           base_out1 = self.base_model(x1)
           out_glb_avg_pool1 = F.avg_pool3d(base_out1, kernel_size=base_out1.size()[2:]).view(base_out1.size()[0],-1)
           linear_out1 = self.dense_1(out_glb_avg_pool1)
           final_out1 = self.dense_2( F.relu(linear_out1))
           y1 = self.sigmoid(final_out1)

           base_out2 = self.base_model(x2)
           out_glb_avg_pool2 = F.avg_pool3d(base_out2, kernel_size=base_out2.size()[2:]).view(base_out2.size()[0],-1)
           linear_out2 = self.dense_1(out_glb_avg_pool2)
           final_out2 = self.dense_2( F.relu(linear_out2))
           y2 = self.sigmoid(final_out2)

           return (y1+y2)/2

        else:
           x1 = x
           base_out1 = self.base_model(x1)
           out_glb_avg_pool1 = F.avg_pool3d(base_out1, kernel_size=base_out1.size()[2:]).view(base_out1.size()[0],-1)
           linear_out1 = self.dense_1(out_glb_avg_pool1)
           final_out1 = self.dense_2( F.relu(linear_out1))
           y1 = self.sigmoid(final_out1)
           return y1
           



        #self.base_out = self.base_model(x)
        #self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        #self.linear_out = self.dense_1(self.out_glb_avg_pool)
        #final_out = self.dense_2( F.relu(self.linear_out))
        #return self.sigmoid(final_out)

class LowRankLinear(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.lowRankLinear = nn.Linear(dim, rank)

    def forward(self, x):
        return self.lowRankLinear(x)

# prepare the 3D model
class Classifier(nn.Module):
    def __init__(self, rank, n_class=1):
        super(Classifier, self).__init__()

        self.dense_1 = nn.Linear(rank, n_class, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(x)
        x = self.dense_1(x)
        x = self.sigmoid(x)
        return x


class MultiHeadNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(MultiHeadNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, 512, bias=True)

        self.kd_GS_thetas = nn.Parameter(torch.ones(7))
        self.ph0 = nn.Linear(512, n_class)
        self.ph1 = nn.Linear(512, n_class)
        self.ph2 = nn.Linear(512, n_class)
        self.ph3 = nn.Linear(512, n_class)
        self.adc = nn.Linear(512, n_class)
        self.t2 = nn.Linear(512, n_class)
        self.other = nn.Linear(512, n_class)

    def forward(self, x, temperature):
        self.base_out = self.base_model(x)
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2( F.relu(self.linear_out))

        kd_GS_thetas = nn.functional.gumbel_softmax(self.kd_GS_thetas, temperature)
        ph0_out = self.ph0(F.relu(final_out))
        ph1_out = self.ph1(F.relu(final_out))
        ph2_out = self.ph2(F.relu(final_out))
        ph3_out = self.ph3(F.relu(final_out))
        adc_out = self.adc(F.relu(final_out))
        t2_out = self.t2(F.relu(final_out))
        other_out = self.other(F.relu(final_out))
        return ph0_out*kd_GS_thetas[0] + ph1_out*kd_GS_thetas[1] + ph2_out*kd_GS_thetas[2] + ph3_out*kd_GS_thetas[3] + adc_out*kd_GS_thetas[4] + t2_out*kd_GS_thetas[5] + other_out*kd_GS_thetas[6]


# prepare the 3D model
class JointHeadNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(JointHeadNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)

        self.ph0 = nn.Linear(1024, n_class)
        self.ph1 = nn.Linear(1024, n_class)
        self.ph2 = nn.Linear(1024, n_class)
        self.ph3 = nn.Linear(1024, n_class)
        self.adc = nn.Linear(1024, n_class)
        self.t2 = nn.Linear(1024, n_class)
        self.other = nn.Linear(1024, n_class)

    def forward(self, x, series_label, temperature):
        self.base_out = self.base_model(x)
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)

        series_label = F.one_hot(series_label, num_classes=7)
        ph0_out = self.ph0(F.relu(self.linear_out))
        ph1_out = self.ph1(F.relu(self.linear_out))
        ph2_out = self.ph2(F.relu(self.linear_out))
        ph3_out = self.ph3(F.relu(self.linear_out))
        adc_out = self.adc(F.relu(self.linear_out))
        t2_out = self.t2(F.relu(self.linear_out))
        other_out = self.other(F.relu(self.linear_out))
        final = torch.cat((ph0_out[None], ph1_out[None], ph2_out[None], ph3_out[None], adc_out[None], t2_out[None], other_out[None]))

        series_label = series_label.permute(1,0)
        final = final[:,:,0] * series_label
        return final.sum(0)[:,None]


class WeightedUnet3D(nn.Module):
      def __init__(self, base_model, n_class=1):
          super(WeightedUnet3D, self).__init__()

          self.base_model = base_model
          
          self.conv = nn.Conv2d(112, 112, 1)
          self.softmax = nn.Softmax(dim=1)
          
          self.dense_1 = nn.Linear(512, 1024, bias=True)
          self.dense_2 = nn.Linear(1024, n_class, bias=True)

      def forward(self, x):
          x = x.permute(0,1,4,2,3)
          out0 = self.conv(x[:,0])
          out1 = self.conv(x[:,1])
          out2 = self.conv(x[:,2])
          out3 = self.conv(x[:,3])
          out4 = self.conv(x[:,4])
          out5 = self.conv(x[:,5])
          out6 = self.conv(x[:,6])
          out = torch.cat((out0[:,None], out1[:,None], out2[:,None], out3[:,None], out4[:,None], out5[:,None], out6[:,None]), dim=1)
          out_soft = self.softmax(out)
          out = out * out_soft
          out = out.sum((1))
          out = out.permute(0,2,3,1)

          self.base_out = self.base_model(out[:,None])
          self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
          self.linear_out = self.dense_1(self.out_glb_avg_pool)
          final_out = self.dense_2( F.relu(self.linear_out))
          return final_out



class SiameseWeightedUnet3D(nn.Module):
      def __init__(self, base_model, n_class=1):
          super(SiameseWeightedUnet3D, self).__init__()

          self.base_model = base_model

          self.conv = nn.Conv2d(128, 128, 1)
          self.softmax = nn.Softmax(dim=1)

          self.dense_1 = nn.Linear(512, 1024, bias=True)
          self.dense_2 = nn.Linear(1024, 128, bias=True)
          self.dense_3 = nn.Linear(128, 2, bias=True)

      def forward(self, x_pre, x_post):
          x_pre = x_pre.permute(0,1,4,2,3)
          out0 = self.conv(x_pre[:,0])
          out1 = self.conv(x_pre[:,1])
          out2 = self.conv(x_pre[:,2])
          out3 = self.conv(x_pre[:,3])
          out4 = self.conv(x_pre[:,4])
          out5 = self.conv(x_pre[:,5])
          out6 = self.conv(x_pre[:,6])
          out = torch.cat((out0[:,None], out1[:,None], out2[:,None], out3[:,None], out4[:,None], out5[:,None], out6[:,None]), dim=1)
          out_soft = self.softmax(out)
          out = out * out_soft
          out = out.sum((1))
          out_pre = out.permute(0,2,3,1)

          self.base_out = self.base_model(out_pre[:,None])

          self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
          self.linear_out = self.dense_1(self.out_glb_avg_pool)
          linear_out2 = self.dense_2( F.relu(self.linear_out))
          pre_out = self.dense_3(F.relu(linear_out2))

          x_post = x_post.permute(0,1,4,2,3)
          out0 = self.conv(x_post[:,0])
          out1 = self.conv(x_post[:,1])
          out2 = self.conv(x_post[:,2])
          out3 = self.conv(x_post[:,3])
          out4 = self.conv(x_post[:,4])
          out5 = self.conv(x_post[:,5])
          out6 = self.conv(x_post[:,6])
          out = torch.cat((out0[:,None], out1[:,None], out2[:,None], out3[:,None], out4[:,None], out5[:,None], out6[:,None]), dim=1)
          out_soft = self.softmax(out)
          out = out * out_soft
          out = out.sum((1))
          out_post = out.permute(0,2,3,1)

          self.base_out = self.base_model(out_post[:,None])
          self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
          self.linear_out = self.dense_1(self.out_glb_avg_pool)
          linear_out2 = self.dense_2( F.relu(self.linear_out))
          post_out = self.dense_3(F.relu(linear_out2))
          return pre_out, post_out


class SiameseWeightedUnet3DGradCam(nn.Module):
      def __init__(self, base_model, n_class=1):
          super(SiameseWeightedUnet3DGradCam, self).__init__()

          self.base_model = base_model

          self.conv = nn.Conv2d(112, 112, 1)
          self.softmax = nn.Softmax(dim=1)

          self.dense_1 = nn.Linear(512, 1024, bias=True)
          self.dense_2 = nn.Linear(1024, 128, bias=True)
          self.dense_3 = nn.Linear(128, 2, bias=True)


      def forward(self, x_pre):
          x_pre = x_pre.permute(0,1,4,2,3)
          out0 = self.conv(x_pre[:,0])
          out1 = self.conv(x_pre[:,1])
          out2 = self.conv(x_pre[:,2])
          out3 = self.conv(x_pre[:,3])
          out4 = self.conv(x_pre[:,4])
          out5 = self.conv(x_pre[:,5])
          out6 = self.conv(x_pre[:,6])
          out = torch.cat((out0[:,None], out1[:,None], out2[:,None], out3[:,None], out4[:,None], out5[:,None], out6[:,None]), dim=1)
          out_soft = self.softmax(out)
          
          out = out * out_soft
          out = out.sum((1))
          out_pre = out.permute(0,2,3,1)
          
          self.base_out = self.base_model(out_pre[:,None])
          self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
          #self.linear_out = self.dense_1(self.out_glb_avg_pool)
          #linear_out2 = self.dense_2( F.relu(self.linear_out))
          #pre_out = self.dense_3(F.relu(linear_out2))

          return self.base_out




