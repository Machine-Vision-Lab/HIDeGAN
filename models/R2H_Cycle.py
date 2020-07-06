import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# from .vgg import Vgg16
from data.base_dataset import get_transform
from sklearn.metrics import mean_squared_error,log_loss
import numpy as np
from PIL import Image
from sklearn import preprocessing
import h5py 
import os
from skimage import color
from skimage.feature import canny
from sklearn.metrics.pairwise import cosine_similarity
from util.metrics import psnr, ssim 
from util.util import tensor2hsi,hsi_normalize
import PIL



class R2HCycleModel(BaseModel):
    """
    This class implements the R2HCycle model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG unet_256' U-Net generator,
    a '--netD pixel' discriminator ,
    and a least-square GANs objective ('--gan_mode vanilla').

    This is a build on CycleGAN model proposed by Jun-Yan Zhu et at. CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For H2RCycle, in addition to GAN losses, we use lambda_A, lambda_B, and lambda_identity, lambda_skel_A, lambda_skel_B  for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| 
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| 
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) 
        Skeleton loss (optional): It has 4 components
            Cyclic skeleton loss : lambda_skel_A * ||Y(x) - Y(G_B(G_A(x))||  
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            
            parser.add_argument('--lambda_color', type=float, default='0', help='weight for color loss. MSE Loss between RGB mappings of real and fake image.')
            parser.add_argument('--lambda_vgg_fake', type=float, default='0', help='weight for perceptual loss calculated using vgg. Loss between real and generated fake image')
            parser.add_argument('--lambda_vgg_cycle', type=float, default='0', help='weight for perceptual loss calculated using vgg. Loss between real and reconstructed real image')
            # The intiution is that increasing weight of L1/L2 loss makes the outputs blurred. Trying to increase GAN loss weight may make image perceptually sound.
            parser.add_argument('--lambda_GAN', type=float, default='2', help='weight for GAN loss')
            parser.add_argument('--lambda_edge_cycle', type=float, default='0.5', help='weight for canny edge loss for cycles')
            parser.add_argument('--lambda_edge_fake', type=float, default='0.5', help='weight for canny edge loss for 1 GAN')
        return parser
    

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        
        self.transform_rgb = get_transform(opt)
        
        
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'vgg_fake', 'vgg_cycle']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'vgg_fake', 'vgg_cycle','idt_A','idt_B','edge_fake','edge_cycle']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc_hsi, opt.output_nc_hsi, opt.ngf, opt.netG_A, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc_hsi, opt.input_nc_hsi, opt.ngf, opt.netG_B, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.vgg = Vgg16(requires_grad=False).to(self.device)
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc_hsi, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc_hsi, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device, dtype=torch.float)  # define GAN loss.
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr)#Adam , betas=(opt.beta1, 0.999)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr)#Adam , betas=(opt.beta1, 0.999)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        
        #if self.isTrain:
        #    self.valA,self.valB = self.valData_loader(opt.val_data_path)
    
    def compute_metrics(self):
        valA, valB = self.valData_loader(self.opt.val_data_path)
        n = len(valB)
        #gen_B = self.netG_A(valA)
        p = []
        s = []
        m = []
        for i,item in enumerate(valA):
            img_A = valA[i,:,:,:]
            img_gen_B = self.netG_A(torch.unsqueeze(img_A, 0))
            img_gen_B = tensor2hsi(img_gen_B)
            #img_gen_B = np.einsum('abc -> bca',img_gen_B)
            step = int(np.floor(32/np.shape(img_gen_B)[2]))
            img_B = valB[i,:,:,::step]
            #print(np.shape(img_gen_B))
            #print(np.shape(img_B))
            p.append(psnr(img_gen_B,img_B))
            s.append(ssim(img_gen_B,img_B))
            m.append(mape(img_B, img_gen_B))
        del n, img_gen_B, img_B, valA, valB
        torch.cuda.empty_cache()
        return np.mean(p),np.mean(s),np.mean(m)
        
    def valData_loader(self, path):    
        file_list_A = os.listdir(path+'valA/')
        file_list_B = os.listdir(path+'valB/')
        val_images = []
        data_B = []
        for file in file_list_A:
            if file.endswith('.jpg'):
                d = Image.open(path+'valA/'+file).convert('RGB')
                rgb_transf = self.transform_rgb(d)
                val_images.append(torch.unsqueeze(rgb_transf, 0))
        val_data_A = torch.cat(val_images, 0)
        for file in file_list_B:
            if file.endswith('.hdf5'):
                with h5py.File(path+'valB/'+file, 'r') as f:
                    data = f['hs_data']
                    d = np.array(data)
                    data_B.append(d)
        val_data_B = np.concatenate([arr[np.newaxis] for arr in data_B])
        del file_list_A, file_list_B, val_images, data_B, d, rgb_transf, data
        return val_data_A,val_data_B


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device, dtype=torch.float)
        #changing real B to RGB to compute loss
        self.real_B = input['B' if AtoB else 'A'].to(self.device, dtype=torch.float)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    
    def free_memory(self):
        del self.fake_B, self.fake_A, self.rec_A, self.rec_B
        del self.loss_D_A, self.loss_G_A, self.loss_cycle_A, self.loss_D_B, self.loss_G_B, self.loss_cycle_B
#         del self.loss_color_A, self.loss_color_B, self.loss_vgg_fake, self.loss_vgg_cycle
        
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        del pred_real, loss_D_real, pred_fake, loss_D_fake
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        del fake_B

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        del fake_A

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_GAN = self.opt.lambda_GAN
        lambda_color = self.opt.lambda_color
        lambda_vgg_fake = self.opt.lambda_vgg_fake
        lambda_vgg_cycle = self.opt.lambda_vgg_cycle
        lambda_edge_cycle = self.opt.lambda_edge_cycle
        lambda_edge_fake = self.opt.lambda_edge_fake
        # Identity loss
        self.rgb_real_B = self.hsi2rgb(self.real_B)
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||  
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            self.loss_idt_B = 0
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * lambda_GAN
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * lambda_GAN
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # MSE Loss
        self.mse_loss = torch.nn.MSELoss() 
        
        # Perceptual Loss
        if lambda_vgg_fake > 0:
            self.loss_vgg_fake = self.perceptual_loss()[0] * lambda_vgg_fake
        else:
            self.loss_vgg_fake = 0
        if lambda_vgg_cycle > 0:
            self.loss_vgg_cycle = self.perceptual_loss()[1] * lambda_vgg_cycle
        else:
            self.loss_vgg_cycle = 0
       
        # Color loss
        if lambda_color > 0:
            self.loss_color_A = self.color_loss()[0] * lambda_color
            self.loss_color_B = self.color_loss()[1] * lambda_color
        else:
            self.loss_color_A = 0
            self.loss_color_B = 0
            
        # Edge Loss
        if lambda_edge_fake > 0:
            self.loss_edge_fake = self.edge_loss()[0] * lambda_edge_fake
        else:    
            self.loss_edge_fake = 0
        
        if lambda_edge_cycle > 0:
            self.loss_edge_cycle = self.edge_loss()[1] * lambda_edge_cycle
        else:    
            self.loss_edge_cycle = 0
            
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B  + self.loss_idt_A + self.loss_idt_B + self.loss_edge_fake + self.loss_edge_cycle
        # self.loss_vgg_fake + self.loss_vgg_cycle + self.loss_color_A + self.loss_color_B 
        self.loss_G.backward()

    def hsi2rgb(self, hsi_batch,channels = 'first'):
        """
        Using magic numbers 28, 16, and 8 we can convert hyperspectral image to RGB
        If we magically reduce channels use the following mapping,
        reduction by factor of 4 leading to 8 channels, RGB = 7,4,2
        input : a batch of HSI
        output : a batch of RGB
        """
        R_ = 28
        G_ = 16
        B_ = 8
        ##### MODIFIED CODE BY A ######
        gen_images = []
        if channels == 'first':
            for item in hsi_batch:
                image = item.cpu().float().detach().numpy()
                r = image[R_,:,:]
                r = preprocessing.minmax_scale(r, feature_range=(0,255), axis=0, copy=True)
                g = image[G_,:,:]
                g = preprocessing.minmax_scale(g, feature_range=(0,255), axis=0, copy=True)
                b = image[B_,:,:]
                b = preprocessing.minmax_scale(b, feature_range=(0,255), axis=0, copy=True)
                rgb = np.stack((r,g,b),axis=0).astype('uint8')
                rgb = np.einsum('abc -> bca',rgb)
                image = Image.fromarray(rgb)#.convert('RGB')
                rgb_transf = self.transform_rgb(image)
                gen_images.append(torch.unsqueeze(rgb_transf, 0))
        elif channels == 'last' :
            for item in hsi_batch:
                image = item.cpu().float().detach().numpy()
                print(image.shape)
                r = image[:,:,R_]
                r = preprocessing.minmax_scale(r, feature_range=(0,255), axis=0, copy=True)
                g = image[:,:,G_]
                g = preprocessing.minmax_scale(g, feature_range=(0,255), axis=0, copy=True)
                b = image[:,:,B_]
                b = preprocessing.minmax_scale(b, feature_range=(0,255), axis=0, copy=True)
                rgb = np.stack((r,g,b),axis=0).astype('uint8')
                rgb = np.einsum('abc -> bca',rgb)
                image = Image.fromarray(rgb)#.convert('RGB')
                rgb_transf = self.transform_rgb(image)
                gen_images.append(torch.unsqueeze(rgb_transf, 0))
            
            del image, r, g, b, rgb, rgb_transf
        return torch.cat(gen_images, 0)

    
    def color_loss(self):
        rgb_fake_B = self.hsi2rgb(self.fake_B)
        rgb_real_B = self.hsi2rgb(self.real_B)
        real_A = self.real_A[0]
        fake_A = self.fake_A[0]
        
        lab_fake_B = (color.rgb2lab(np.einsum('abc -> cba',rgb_fake_B[0])))
        lab_real_B = (color.rgb2lab(np.einsum('abc -> cba',rgb_real_B[0])))
        
        lab_real_A = (color.rgb2lab(np.einsum('abc -> cba',real_A.cpu().detach().numpy())))
        lab_fake_A = (color.rgb2lab(np.einsum('abc -> cba',fake_A.cpu().detach().numpy())))
        
#         print(type(lab_real_A))
#         print(lab_real_A.shape)
#         print(type(lab_real_B))
#         print(lab_real_B.shape)
        loss_A = (self.mse_loss(torch.from_numpy(lab_real_A),torch.from_numpy(lab_fake_B)))
        loss_B = (self.mse_loss(torch.from_numpy(lab_real_B),torch.from_numpy(lab_fake_A)))
        
        #a = log_loss(lab_real_A, lab_fake_B)
        #b = log_loss(lab_real_B, lab_fake_A)

        
        #print(loss_A.eval())
        #print(loss_B.eval())
        final_loss = [loss_A, loss_B]
        del rgb_fake_B, rgb_real_B, loss_A, loss_B,lab_fake_B, lab_real_B, lab_real_A, lab_fake_A
        return final_loss
        
        
    def perceptual_loss(self):
        real_A = self.real_A
        fake_A = self.fake_A
        rec_B = self.rec_B
        rec_A = self.rec_A
        
        #print("np shape rgb real b : {0}".format(np.shape(rgb_real_B)))
        rgb_real_B = self.rgb_real_B
        rgb_fake_B = self.hsi2rgb(self.real_B)
        rgb_real_A = self.hsi2rgb(real_A)
        rgb_fake_A = self.hsi2rgb(fake_A)
        rgb_rec_A = self.hsi2rgb(rec_A)
        rgb_rec_B = self.hsi2rgb(rec_B)
        
        features_fake_B = self.vgg(rgb_fake_B.to(self.device))
        features_real_A = self.vgg(rgb_real_A.to(self.device))
        features_real_B = self.vgg(rgb_real_B.to(self.device))
        features_fake_A = self.vgg(rgb_fake_A.to(self.device))
        
        features_rec_A = self.vgg(rgb_rec_A.to(self.device))
        features_rec_B = self.vgg(rgb_rec_B.to(self.device))
        
        mse_loss = self.mse_loss
        
        m0 = mse_loss(features_real_A.relu3_3,features_fake_B.relu3_3)
        m1 = mse_loss(features_real_A.relu4_3,features_fake_B.relu4_3)
        m2 = mse_loss(features_real_B.relu3_3,features_fake_A.relu3_3)
        m3 = mse_loss(features_real_B.relu4_3,features_fake_A.relu4_3)
        
        
        m4 = mse_loss(features_real_A.relu3_3,features_rec_A.relu3_3)
        m5 = mse_loss(features_real_A.relu4_3,features_rec_A.relu4_3)
        m6 = mse_loss(features_real_B.relu3_3,features_rec_B.relu3_3)
        m7 = mse_loss(features_real_B.relu4_3,features_rec_B.relu4_3)
        loss_vgg_fake = (m0 + m1 + m2 + m3)*0.5 
        loss_vgg_cycle = (m4 + m5 + m6 + m7)*0.5 
 
        del m2, m0, m6, m7, m1, m3, m4, m5
        del features_fake_A, features_fake_B, features_real_A, features_real_B, features_rec_A, features_rec_B
        del real_A, fake_A, rec_B, rec_A, rgb_real_B, rgb_fake_B, rgb_rec_B, rgb_real_A, rgb_fake_A, rgb_rec_A, mse_loss
        return loss_vgg_fake, loss_vgg_cycle
        
    def edge_loss(self):
        #normalizing all images to 0-1
#         real_A_hsi = hsi_normalize(tensor2hsi(self.real_A),1,True)
#         fake_B_hsi = hsi_normalize(tensor2hsi(self.fake_B),1,True)
#         real_B_hsi = hsi_normalize(tensor2hsi(self.real_B),1,True)
#         fake_A_hsi = hsi_normalize(tensor2hsi(self.fake_A),1,True)
#         rec_A_hsi = hsi_normalize(tensor2hsi(self.rec_A),1,True)
#         rec_B_hsi = hsi_normalize(tensor2hsi(self.rec_B),1,True)
        
    
            
        #converting to RGB then Grayscale
#         print(real_A_hsi.shape)
#         real_A = color.rgb2gray(self.hsi2rgb(torch.tensor(real_A_hsi),'last'))
#         fake_B = color.rgb2gray(self.hsi2rgb(fake_B_hsi),'last')
#         real_B = color.rgb2gray(self.hsi2rgb(real_B_hsi),'last')
#         fake_A = color.rgb2gray(self.hsi2rgb(fake_A_hsi),'last')
#         rec_A = color.rgb2gray(self.hsi2rgb(rec_A_hsi),'last')
#         rec_B = color.rgb2gray(self.hsi2rgb(rec_B_hsi),'last')
        
    
        #new approach : convert to rgb batches. For i in batch, normalize, compute canny edge and dissimilarity and append. 
        
        
        real_A_batch = (self.hsi2rgb(self.real_A))
        fake_B_batch = (self.hsi2rgb(self.fake_B))
        real_B_batch = (self.hsi2rgb(self.real_B))
        fake_A_batch = (self.hsi2rgb(self.fake_A))
        rec_A_batch = (self.hsi2rgb(self.rec_A))
        rec_B_batch = (self.hsi2rgb(self.rec_B))
        
        batch_size = (real_A_batch).shape[0]
        loss_edge_fake = 0.
        loss_edge_cycle = 0.
        for i in range(batch_size):
            
            real_A = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',real_A_batch[i])),1,True)
            fake_B = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',fake_B_batch[i])),1,True)
            real_B = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',real_B_batch[i])),1,True)
            fake_A = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',fake_A_batch[i])),1,True)
            rec_A = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',rec_A_batch[i])),1,True)
            rec_B = hsi_normalize(color.rgb2gray(np.einsum('abc->bca',rec_B_batch[i])),1,True)
                     

            edges_real_A = canny(real_A)
            edges_fake_B = canny(fake_B)
            edges_real_B = canny(real_B)
            edges_fake_A = canny(fake_A)
            edges_rec_A = canny(rec_A)
            edges_rec_B = canny(rec_B)
#             print(real_A.shape)
# #             im = np.einsum('abc -> bca',edges_real_A)    
#             im = Image.fromarray(edges_real_A)
#             im = im.save("/data/dehazing/Codes/test.jpg")
#             im2 = Image.fromarray(real_A)
#             im2 = im2.save("/data/dehazing/Codes/test2.png")
            # finding dissimilarity
            a = 1 - np.mean(cosine_similarity(edges_real_A, edges_fake_B, dense_output=True))
            b = 1 - np.mean(cosine_similarity(edges_real_B, edges_fake_A, dense_output=True))
            c = 1 - np.mean(cosine_similarity(edges_real_A, edges_rec_A, dense_output=True))
            d = 1 - np.mean(cosine_similarity(edges_real_B, edges_rec_B, dense_output=True))

            loss_edge_fake += (a+b)*0.5 
            loss_edge_cycle += (c+d)*0.5
 
        del a,b,c,d
        del edges_fake_A, edges_fake_B, edges_real_A, edges_real_B, edges_rec_A, edges_rec_B
        del real_A, fake_A, rec_B, rec_A, real_B, fake_B, real_A_batch, fake_A_batch, rec_B_batch, rec_A_batch, real_B_batch, fake_B_batch
        return loss_edge_fake, loss_edge_cycle
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        
    def get_min(self, data):
        return np.min(np.min(data, axis=0), axis=0).min()

    def get_max(self, data):
        return np.max(np.max(data, axis=0), axis=0).max()


