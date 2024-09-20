import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
from helper import *
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import os




class FBPinns2D:
    def __init__(self, n_int_,w_list, n_layers,n_neurons, n_subdomain=(15,15), overlap_len=0.3):
        self.n_int = n_int_
        self.n_sub = n_subdomain
        self.w_list = w_list
        self.w_max =  torch.Tensor(self.w_list).max()
        self.n_layer =  n_layers
        self.n_neuron = n_neurons
        self.overlap_len =overlap_len
        self.device = torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")###  # #



        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-2*np.pi, 2*np.pi],
                                            [-2*np.pi, 2*np.pi]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        # self.lambda_u = 10

        # List of NNs to approximate the solution over the subdomains
        self.nn_list = self.get_nnlist()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.cen_list, self.wid_list, self.len_list, self.four_list ,self.area= self.get_subdomain_list()
    
        

     ################################################################################################
    # Function to return a list of NNs over different subdomains
    def get_nnlist(self):
        nn_list = [[0 for i in range(self.n_sub[1])] for j in range(self.n_sub[0])]
        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]):
                nn_list[i][j] = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                                n_hidden_layers=self.n_layer,
                                                neurons=self.n_neuron,
                                                regularization_param=0.,
                                                regularization_exp=2.,
                                                retrain_seed=42).to(self.device)
        return nn_list
    
    def load_nnlist(self,net_file):
        net = torch.load('net_list.pkl')
        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]):
                self.nn_list[i][j].load_state_dict(net[i][j])
   

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # No initial condions in this problem
    def initial_condition(self, x):
        pass

    # Exact solution for the heat equation ut = u_xx with the IC above
    def exact_solution(self, inputs):
        u = torch.sin(self.w_list[0]*inputs[:,0])*(1/self.w_list[0])+torch.sin(self.w_list[0]*inputs[:,1])*(1/self.w_list[0])
        return u.reshape(-1,1)



    


    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        pass
    
    def add_spatial_boundary_points(self):


        return torch.Tensor([[0]]),  torch.Tensor([[0]])


    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        
        return input_int, output_int
    
    # Function decomposing the domain in one certain dim
    def get_area(self,dim):
        sub_len  = (self.domain_extrema[dim][1]-self.domain_extrema[dim][0])/self.n_sub[dim]
        unif_bnd = torch.linspace(self.domain_extrema[0][0],self.domain_extrema[0][1],steps=self.n_sub[dim]+1)
        bnd_plus = unif_bnd + self.overlap_len/2
        bnd_min = unif_bnd - self.overlap_len/2
        unif_area = torch.Tensor([[unif_bnd[i],unif_bnd[i+1]] for i in range(self.n_sub[dim])])
        overlap_area = torch.Tensor([[bnd_min[i+1],bnd_plus[i+1]] for i in range(self.n_sub[dim]-1)]) # Tensor (n_sub-1) * 2
        center_area = torch.Tensor([[bnd_plus[i],bnd_min[i+1]] for i in range(self.n_sub[dim])]) # Tensor n_sub * 2
        #The firs and last term of center_area should be confined in the domain
        center_area[0][0]  = self.domain_extrema[0][0]
        center_area[-1][1] = self.domain_extrema[0][1] 
        subdomain_area = torch.Tensor([[bnd_min[i],bnd_plus[i+1]] for i in range(self.n_sub[dim])])
        #The firs and last term of subdomain area should be confined in the domain
        subdomain_area[0][0] = self.domain_extrema[0][0]
        subdomain_area[-1][1] = self.domain_extrema[0][1]

        return {'uni':unif_area, 'ovl':overlap_area, 'cen':center_area, 'sub':subdomain_area}
    

    
    def get_subdomain_list(self):
        area = {0:self.get_area(0),1:self.get_area(1)}

        input_int, output_int = self.add_interior_points()         
        center_list = [[0 for i in range(self.n_sub[1])] for i in range(self.n_sub[0])]
        ovl_wid_list = [[0 for i in range(self.n_sub[1]-1)] for i in range(self.n_sub[0])]
        ovl_len_list =[[0 for i in range(self.n_sub[1])] for i in range(self.n_sub[0]-1)]  # 2 overlapped horizontal
        ovl_4_list = [[0 for i in range(self.n_sub[1]-1)] for i in range(self.n_sub[0]-1)]


        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]):
                ind = torch.where(get_ind(input_int,area[0]['cen'][i][0],area[0]['cen'][i][1],area[1]['cen'][j][0],area[1]['cen'][j][1]))
                center_list[i][j] = input_int[ind].reshape(-1,2).to(self.device)
        
        # wid shu
        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]-1):
                ind = torch.where(get_ind(input_int,area[0]['cen'][i][0],area[0]['cen'][i][1],area[1]['ovl'][j][0],area[1]['ovl'][j][1]))
                ovl_wid_list[i][j] = input_int[ind].reshape(-1,2).to(self.device)
        
        # len heng
        for i in range(self.n_sub[0]-1):
            for j in range(self.n_sub[1]):
                ind = torch.where(get_ind(input_int,area[0]['ovl'][i][0],area[0]['ovl'][i][1],area[1]['cen'][j][0],area[1]['cen'][j][1]))
                ovl_len_list[i][j] = input_int[ind].reshape(-1,2).to(self.device)

        for i in range(self.n_sub[0]-1):
            for j in range(self.n_sub[1]-1):
                ind = torch.where(get_ind(input_int,area[0]['ovl'][i][0],area[0]['ovl'][i][1],area[1]['ovl'][j][0],area[1]['ovl'][j][1]))
                ovl_4_list[i][j] = input_int[ind].reshape(-1,2).to(self.device)
    



        return center_list,ovl_wid_list,ovl_len_list,ovl_4_list, area
        
    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        pass

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        pass
    ################################################################################################
    '''
    The Loss is defined as L = 1/Np * sigma(L2loss(D(u,lambda),f(x))
    '''
    def f_x(self,input_int):
        sum = torch.cos(self.w_list[0]*input_int[:,0])+torch.cos(self.w_list[0]*input_int[:,1])

        return sum
    

    def normalize2d(self,x,subdomain0,subdomain1):
        #given the subdomain the input is located in, normalize the input into [-1,1]
        '''
        subdomain: Tensor (2,)
        '''

        k = (1-(-1))/torch.Tensor([(subdomain0[1]-subdomain0[0]),(subdomain1[1]-subdomain1[0])]).to(self.device)
        #breakpoint() #TODO check the shap
        result  = -1 + k*(x-torch.Tensor([subdomain0[0],subdomain1[0]]).to(self.device))
        assert(result.shape[1]==2)

        return result

    def get_center_residual(self, model, input_int, subi,subj):
        '''
        subdomain determines the scale of the normalization for the input variable
        '''
        input_int.requires_grad = True
        u = unnormalize2d(model(self.normalize2d(input_int, subi,subj)))
        w_max =  torch.Tensor(self.w_list).max()
        u = (1/w_max)*torch.sin(w_max*input_int[:,1]).reshape(-1,1)+torch.tanh(w_max*input_int[:,0]).reshape(-1,1)*u
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        D_u = grad_u[:,0]+grad_u[:,1]
        out = self.f_x(input_int)

        assert (D_u.shape == out.shape)
        residual = l2_loss(D_u,out)
        with torch.no_grad():
            error = l1_loss(self.exact_solution(input_int),u)

        #residual = l2_loss(self.exact_solution(input_int),u)
        return residual,error

    def get_ovl_residual(self, input_int,model_list,subi_list,subj_list,unii_list,unij_list):

        assert(len(model_list)==len(subi_list)==len(subj_list)==len(unii_list)==len(unij_list))
        input_int.requires_grad = True

        u_sum = 0
        for (model,subi,subj,unii,unij) in zip(model_list,subi_list,subj_list,unii_list,unij_list):

            u_sum += unnormalize2d( model(self.normalize2d(input_int,subi,subj)))*window2d(input_int,unii,unij)

   
        w_max =  torch.Tensor(self.w_list).max()

        u = (1/w_max)*torch.sin(w_max*input_int[:,1]).reshape(-1,1)+torch.tanh(w_max*input_int[:,0]).reshape(-1,1)*u_sum
        
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        D_u = grad_u[:,0]+grad_u[:,1]
        out = self.f_x(input_int)
        assert (D_u.shape == out.shape)

        residual = l2_loss(D_u,out)
        with torch.no_grad():
            error = l1_loss(self.exact_solution(input_int),u)


        return residual,error
    

    # Function to compute the mean of the physics loss of the center domains and the overlapped domains
    def compute_loss(self, verbose=True):
        r_cent_list = []
        r_ovl_list = []
        err_cent_list = []
        err_ovl_list = []
        for i in range(self.n_sub[0]): 
            for j in range(self.n_sub[1]):
                r_cent,err_cent = self.get_center_residual(self.nn_list[i][j],self.cen_list[i][j],self.area[0]['sub'][i],self.area[1]['sub'][j])
                r_cent_list.append(r_cent)
                err_cent_list.append(err_cent)
                if j<=self.n_sub[1]-2:
                    r_ovl_len, err_ovl_len  = self.get_ovl_residual(input_int = self.wid_list[i][j],
                                                model_list=[self.nn_list[i][j],self.nn_list[i][j+1]],
                                                subi_list=[self.area[0]['sub'][i],self.area[0]['sub'][i]],
                                                subj_list=[self.area[1]['sub'][j],self.area[1]['sub'][j+1]],
                                                unii_list=[self.area[0]['uni'][i],self.area[0]['uni'][i]],
                                                unij_list=[self.area[1]['uni'][j],self.area[1]['uni'][j+1]])
                    r_ovl_list.append(r_ovl_len)
                    err_ovl_list.append(err_ovl_len)

                # ovl_2_len and ovl_4
                if i <= self.n_sub[0]-2:

                    r_ovl_len, err_ovl_len  = self.get_ovl_residual(input_int = self.len_list[i][j],
                                                model_list=[self.nn_list[i][j],self.nn_list[i+1][j]],
                                                subi_list=[self.area[0]['sub'][i],self.area[0]['sub'][i+1]],
                                                subj_list=[self.area[1]['sub'][j],self.area[1]['sub'][j]],
                                                unii_list=[self.area[0]['uni'][i],self.area[0]['uni'][i+1]],
                                                unij_list=[self.area[1]['uni'][j],self.area[1]['uni'][j]])
                    r_ovl_list.append(r_ovl_len)
                    err_ovl_list.append(err_ovl_len)

                    if j<=self.n_sub[1]-2:
                        #oder is 1,2
                        #        3,4
                        r_ovl_4, err_ovl_4  = self.get_ovl_residual(input_int = self.four_list[i][j],
                                                model_list = [self.nn_list[i][j],self.nn_list[i][j+1],self.nn_list[i+1][j],self.nn_list[i+1][j+1]],
                                                subi_list  = [self.area[0]['sub'][i],self.area[0]['sub'][i],self.area[0]['sub'][i+1],self.area[0]['sub'][i+1]],
                                                subj_list  = [self.area[1]['sub'][j],self.area[1]['sub'][j+1],self.area[1]['sub'][j],self.area[1]['sub'][j+1]],
                                                unii_list  = [self.area[0]['uni'][i],self.area[0]['uni'][i],self.area[0]['uni'][i+1],self.area[0]['uni'][i+1]],
                                                unij_list  = [self.area[1]['uni'][j],self.area[1]['uni'][j+1],self.area[1]['uni'][j],self.area[1]['uni'][j+1]])
                        r_ovl_list.append(r_ovl_4)
                        err_ovl_list.append(err_ovl_4)

    

        res = torch.concat((r_cent_list+r_ovl_list),0) #+r_ovl_list

        err = torch.log10(torch.mean(torch.concat((err_cent_list+err_ovl_list),0)))

        loss =  torch.log10(torch.mean(res))  # torch.Tensor() method creates a new tensor without gradients

        if verbose: print("Total loss: ", round(loss.item(), 4),"| L1 error: ", round(err.item(), 4),)

        return loss, err
    
    def get_ovlres_mp(self, input_int,model,subi,subj,unii,unij):


        input_int.requires_grad = True


        u = unnormalize2d( model(self.normalize2d(input_int,subi,subj)))*window2d(input_int,unii,unij)
        u_sep = torch.tanh(self.w_max*input_int[:,0]).reshape(-1,1)*u
        grad_u = torch.autograd.grad(u_sep.sum(), input_int, create_graph=True)[0]


        return grad_u
    
    def compute_forward_kernel(self,ind):
        # i: index of the domain number

        i =  ind //self.n_sub[1]
        j =  ind % self.n_sub[1]
        subi =  self.area[0]['sub'][i]
        subj =  self.area[1]['sub'][j]
        unii =  self.area[0]['uni'][i]
        unij =  self.area[1]['uni'][j]
        #cent res
        res_cent, _ = self.get_center_residual(self.nn_list[i][j],self.cen_list[i][j],subi,subj)
        #overlap
      
        def ovl2_res(u,d,l,r): #TODO ,subi,subj,unii,unij
            if u == 0:
                up = 0
            else:
                up =  self.get_ovlres_mp(self.len_list[i-1][j],self.nn_list[i][j],subi,subj,unii,unij)

            if d == 0:
                down = 0
            else:
                down  =  self.get_ovlres_mp(self.len_list[i][j],self.nn_list[i][j],subi,subj,unii,unij)

            if l == 0:
                left = 0
            else:
                left  =  self.get_ovlres_mp(self.wid_list[i][j-1],self.nn_list[i][j],subi,subj,unii,unij)

            if r == 0:
                right = 0
            else:
                right =  self.get_ovlres_mp(self.wid_list[i][j],self.nn_list[i][j],subi,subj,unii,unij)

            return up,down,left,right
        
        def ovl4_res(u,d,l,r): 
            #position  u d
            #          l r
            if u == 0:
                up = 0
            else:
                up =  self.get_ovlres_mp(self.four_list[i-1][j-1],self.nn_list[i][j],subi,subj,unii,unij)

            if d == 0:
                down = 0
            else:
               
                down =  self.get_ovlres_mp(self.four_list[i-1][j],self.nn_list[i][j],subi,subj,unii,unij)

            if l == 0:
                left = 0
            else:
                left  = self.get_ovlres_mp(self.four_list[i][j-1],self.nn_list[i][j],subi,subj,unii,unij)

            if r == 0:
                right = 0
            else:
                right =  self.get_ovlres_mp(self.four_list[i][j],self.nn_list[i][j],subi,subj,unii,unij)

            return up,down,left,right

        # return up left down right #或者改成concat
        if (i>0) and (i < 14) and (j > 0) and (j < 14):
            up,down,left,right =  ovl2_res(1,1,1,1)
            u,d,l,r = ovl4_res(1,1,1,1)
        
        elif ((i,j) == (0,0)) or ((i,j) == (0,14)) or ((i,j) == (14,0)) or ((i,j) == (14,14)):
            if (i,j) == (0,0):
                up,down,left,right =  ovl2_res(0,1,0,1)
                u,d,l,r = ovl4_res(0,0,0,1)

            
            if ((i,j) == (0,14)):
                up,down,left,right =  ovl2_res(0,1,1,0)
                u,d,l,r = ovl4_res(0,0,1,0)

            
            if ((i,j) == (14,0)):
                up,down,left,right =  ovl2_res(1,0,0,1)
                u,d,l,r = ovl4_res(0,1,0,0)
      
            
            if ((i,j) == (14,14)):
                up,down,left,right =  ovl2_res(1,0,1,0)
                u,d,l,r = ovl4_res(1,0,0,0)

        else:
            if (i == 0):
                up,down,left,right =  ovl2_res(0,1,1,1)
                u,d,l,r = ovl4_res(0,0,1,1)

            
            if (i == 14):
                up,down,left,right =  ovl2_res(1,0,1,1)
                u,d,l,r = ovl4_res(1,1,0,0)


            if (j == 0):
                up,down,left,right =  ovl2_res(1,1,0,1)
                u,d,l,r = ovl4_res(0,1,0,1)
           
            if (j == 14):
                up,down,left,right =  ovl2_res(1,1,1,0)
                u,d,l,r = ovl4_res(1,0,1,0)
  
        return (res_cent, up, down, left, right,u,d,l,r)
    



    

    def du_1(self,input_int):
        n = input_int.shape[0]
        du_x1 = torch.zeros_like(input_int[:,0].reshape(-1,1))
        du_x2 = torch.cos(self.w_max*input_int[:,1].reshape(-1,1)) 
        return torch.cat((du_x1,du_x2),dim=1)
    
    def ovl2_res(self,input_int,du_ex,du_int):
                    
        du = du_ex+du_int+self.du_1(input_int)
        D_u = du[:,0]+du[:,1]
        out = self.f_x(input_int)
        return l2_loss(D_u,out)
    
    def ovl4_res(self,input_int,u,d,l,r):
                    
        du = u+d+l+r+self.du_1(input_int)
        D_u = du[:,0]+du[:,1]
        out = self.f_x(input_int)
        return l2_loss(D_u,out)
    
    def compute_ovl_kernel(self,ind):
        # i: index of the domain number
        i =  ind //self.n_sub[1]
        j =  ind % self.n_sub[1]
        res_cen = self.par[i][j][0]


        

        def compute_ovl2_res(u,d,l,r): #TODO ,subi,subj,unii,unij
            if u == 0:
                up = torch.Tensor([0])
            else:
                up = self.ovl2_res(self.len_list[i-1][j],self.par[i-1][j][2],self.par[i][j][1])

            if d == 0:
                down = torch.Tensor([0])
            else:
                down  =  self.ovl2_res(self.len_list[i][j],self.par[i+1][j][1],self.par[i][j][2])

            if l == 0:
                left = torch.Tensor([0])
            else:
                left  =  self.ovl2_res(self.wid_list[i][j-1],self.par[i][j-1][4],self.par[i][j][3])

            if r == 0:
                right = torch.Tensor([0])
            else:
                right =  self.ovl2_res(self.wid_list[i][j],self.par[i][j+1][3],self.par[i][j][4])

            return up,down,left,right
        
        def compute_ovl4_res(u,d,l,r): #TODO ,subi,subj,unii,unij
            
   
            #position  u d
            #          l r
            if u == 0:
                up = torch.Tensor([0])
            else:
                up =  self.ovl4_res(self.four_list[i-1][j-1],self.par[i-1][j-1][8],self.par[i-1][j][7],self.par[i][j-1][6],self.par[i][j][5])

            if d == 0:
                down = torch.Tensor([0])
            else:
                down =  self.ovl4_res(self.four_list[i-1][j],self.par[i-1][j][8],self.par[i-1][j+1][7],self.par[i][j][6],self.par[i][j+1][5])

            if l == 0:
                left = torch.Tensor([0])
            else:
                left  = self.ovl4_res(self.four_list[i][j-1],self.par[i][j-1][8],self.par[i][j][7],self.par[i+1][j-1][6],self.par[i+1][j][5])

            if r == 0:
                right = torch.Tensor([0])
            else:
                right =  self.ovl4_res(self.four_list[i][j],self.par[i][j][8],self.par[i][j+1][7],self.par[i+1][j][6],self.par[i+1][j+1][5])

            return up,down,left,right
        


        # return up left down right #或者改成concat
        if (i>0) and (i < 14) and (j > 0) and (j < 14):
            up,down,left,right =  compute_ovl2_res(1,1,1,1)
            u,d,l,r = compute_ovl4_res(1,1,1,1)
        
        elif ((i,j) == (0,0)) or ((i,j) == (0,14)) or ((i,j) == (14,0)) or ((i,j) == (14,14)):
            if (i,j) == (0,0):
                up,down,left,right =  compute_ovl2_res(0,1,0,1)
                u,d,l,r = compute_ovl4_res(0,0,0,1)

            
            if ((i,j) == (0,14)):
                up,down,left,right =  compute_ovl2_res(0,1,1,0)
                u,d,l,r = compute_ovl4_res(0,0,1,0)

            
            if ((i,j) == (14,0)):
                up,down,left,right =  compute_ovl2_res(1,0,0,1)
                u,d,l,r = compute_ovl4_res(0,1,0,0)
      
            
            if ((i,j) == (14,14)):
                up,down,left,right =  compute_ovl2_res(1,0,1,0)
                u,d,l,r = compute_ovl4_res(1,0,0,0)

        else:
            if (i == 0):
                up,down,left,right =  compute_ovl2_res(0,1,1,1)
                u,d,l,r = compute_ovl4_res(0,0,1,1)

            
            if (i == 14):
                up,down,left,right =  compute_ovl2_res(1,0,1,1)
                u,d,l,r = compute_ovl4_res(1,1,0,0)


            if (j == 0):
                up,down,left,right =  compute_ovl2_res(1,1,0,1)
                u,d,l,r = compute_ovl4_res(0,1,0,1)

        
            if j == 14:
                up,down,left,right =  compute_ovl2_res(1,1,1,0)
                u,d,l,r = compute_ovl4_res(1,0,1,0)
  
        res_total = torch.cat((res_cen,up,down,left,right,u,d,l,r),0)

        
        return res_total
        
    def compute_ovl(self,ind_row):
        res = []
        for i in range(self.n_sub[1]):
            ind =  ind_row*self.n_sub[0] +i
            res.append(self.compute_ovl_kernel(ind))
        return res

    def forward(self,ind_row):
       
        res = []

        for i in range(self.n_sub[1]):
            ind =  ind_row*self.n_sub[0] +i
            res.append(self.compute_forward_kernel(ind))
        return res
    
    def job(self,i):
        return i
    
    def fit_mp(self, num_epochs, optimizer, verbose=True):
        history = list()
        hist_error = list()
        
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: 
                if epoch % 1 == 0:
                    print("################################ ", epoch, " ################################")

                #forward
                # res = []
                # for i in range(15):
                #     res.append(self.forward(i))

                # breakpoint()
                #torch.autograd.set_detect_anomaly(True)
                optimizer.zero_grad()
                t_list=[]
                res = []
                for i in range(15):
                    #res.append(self.forward(i))
                    t = MyThread(self.forward, (i,))      
                    t.start()
                    t_list.append(t)
                for i in t_list:
                    i.join()
                
                for i in t_list:
                    res.append(i.getResult())

                self.par= res
                res_list = []
                t_list2= []
                for i in range(15):
                    #res_list.append(self.compute_ovl(i))

                    t = MyThread(self.compute_ovl,(i,))
                    t.start()
                    t_list2.append(t)
                
                for i in t_list2:
                    i.join()

                for i in t_list2:
                    res_list.append(i.getResult())


     
                loss =  torch.log10(torch.mean(torch.cat(listto1d(res_list),0)))
                loss.backward()
                optimizer.step()
                print("Total loss: ", round(loss.item(), 4))
                
          
                # res_list = pool.starmap(self.compute_ovl,[(i,optimizer_list[i]) for i in range(self.n_sub[0])])
                # res_total = torch.log10(torch.mean(torch.cat(listto1d(res_list),0)))
                

                history.append(loss.item())
                #hist_error.append(err.item())

        print('Final Loss: ', history[-1])
        #print('Final L1 error: ', hist_error[-1])

        return history#, hist_error

    

if __name__ == '__main__':
    
    train = True
    n_int = 900*900
    n_layers = 2
    n_neurons = 16
    ovl=0.3
    fbpinn = FBPinns2D(n_int_=n_int,w_list=[15],n_layers=n_layers,n_neurons=n_neurons,n_subdomain=(15,15),overlap_len=ovl)
    
    n_epochs = 20


    if train == True:
        # para = []
        # for i in fbpinn.nn_list:
        #     para += list(i.parameters())
        
        
        # optimizer_LBFGS = optim.LBFGS(para,
        #                             lr=float(0.1),
        #                             max_iter=50000, 
        #                             max_eval=50000,
        #                             history_size=150,
        #                             line_search_fn="strong_wolfe",
        #                             tolerance_change=1.0 * np.finfo(float).eps)

       
        #optimizer_ADAM_list = listto2d([optim.Adam(listto1d(fbpinn.nn_list)[i].parameters(),lr=float(0.001)) for i in range(15*15)],15,15)
        #optimizer_ADAM_list = [optim.Adam(listto1d(fbpinn.nn_list)[0].parameters(),lr=float(0.001))]
        para_adam = []
        for i in range(fbpinn.n_sub[0]):
            for j in range(fbpinn.n_sub[1]):
                para_adam.append({'params':fbpinn.nn_list[i][j].parameters()})
        optimizer_ADAM = optim.Adam(para_adam,lr=float(0.001))
        start  =time.time()
        hist  = fbpinn.fit_mp(n_epochs,
                        optimizer_ADAM,
                        verbose=True)
        end = time.time()
        print('The training time is {} s'.format(end-start))

        #save
        # net = [[0 for i in range(fbpinn.n_sub[1])] for j in range(fbpinn.n_sub[0])]
        # for i in range(fbpinn.n_sub[0]):
        #     for j in range(fbpinn.n_sub[1]):
        #         net[i][j] =  fbpinn.nn_list[i][j].state_dict()
        # torch.save(net,'net_list.pkl')
        # plot the training loss
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.savefig('trainloss.png')

        #L1 errora
        # plt.figure(dpi=150)
        # plt.grid(True, which="both", ls=":")
        # plt.plot(np.arange(1, len(hist_error) + 1), hist_error, label="L1 error")
        # plt.xscale("log")
        # plt.legend()
        # plt.savefig('L1_error.png')
        
        # #pred the training predictions
        # x = torch.linspace(fbpinn.domain_extrema[0][0],fbpinn.domain_extrema[0][1], 2000).reshape(-1,1)
        # y_exact = fbpinn.exact_solution(x)
        # y_pred_cen,y_pred_ovl, x_cent,x_ovl  = fbpinn.predict(x)
        # plt.figure(figsize=(30,4.8))
        # plt.grid(True, which="both", ls=":")
 
        # plt.scatter(x_cent, fbpinn.exact_solution(x_cent), label="Ground Truth",s=10)
        # plt.scatter(x_cent, y_pred_cen, label="Network Prediction",s=10)
        # plt.scatter(x_ovl, fbpinn.exact_solution(x_ovl), label="Ground Truth",s=10)
        # plt.scatter(x_ovl, y_pred_ovl, label="Network Prediction",s=10)
        # plt.xlabel("x")
        # plt.ylabel("u")
        # plt.legend()
        # plt.savefig('pred.png')
    else:
        fbpinn.plot_subarea()
