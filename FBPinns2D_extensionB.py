import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import sys
sys.path.append('..')
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
from helper import *
import matplotlib.pyplot as plt
import time

# All scheduled implementation of FBPinns over the problemset:
# du/dx = w1*cos(w1*x)+w2*cos(w2*x)

class FBPinns2D:
    def __init__(self, n_int_,w_list, n_layers,n_neurons, n_subdomain=(15,15), overlap_len=0.3):
        self.n_int = n_int_
        self.n_sub = n_subdomain
        self.w_list = w_list
        self.n_layer =  n_layers
        self.n_neuron = n_neurons
        self.overlap_len =overlap_len
        self.device = torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")#  # #



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
        #self.cen_list, self.wid_list, self.len_list, self.four_list ,self.area= self.get_subdomain_list()
        self.cen_list, self.wid_list, self.len_list, self.four_list ,self.area= self.get_pred_subdomain_list()
    
        

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
    
    def get_pred_subdomain_list(self):
        area = {0:self.get_area(0),1:self.get_area(1)}

        inputs = self.soboleng.draw(100000)
        input_int = self.convert(inputs)       
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

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        hist_error = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: 
                if epoch % 200 == 0:
                    print("################################ ", epoch, " ################################")

 
            def closure():
                optimizer.zero_grad()
                loss,err = self.compute_loss(verbose=verbose)
                loss.backward()

                history.append(loss.item())
                hist_error.append(err.item())
                return loss

            optimizer.step(closure=closure)

           
            if epoch % 500 == 0:
                net_tmp = [[0 for i in range(fbpinn.n_sub[1])] for j in range(fbpinn.n_sub[0])]

                for i in range(fbpinn.n_sub[0]):
                    for j in range(fbpinn.n_sub[1]):
                        net_tmp[i][j] =  self.nn_list[i][j].state_dict()
                
                torch.save(net_tmp,'net_list_ep{}.pth'.format(epoch))
                torch.save(history,"train_hist_{}.pth".format(epoch))
                torch.save(hist_error,'hist_error_{}.pth'.format(epoch))
        

        print('Final Loss: ', history[-1])
        print('Final L1 error: ', hist_error[-1])

        return history, hist_error
    
    def pred_center(self, model, input_int, subi,subj):
        '''
        subdomain determines the scale of the normalization for the input variable
        '''

        u = unnormalize2d(model(self.normalize2d(input_int, subi,subj)))
        w_max =  torch.Tensor(self.w_list).max()
        u = (1/w_max)*torch.sin(w_max*input_int[:,1]).reshape(-1,1)+torch.tanh(w_max*input_int[:,0]).reshape(-1,1)*u


        
        return torch.cat((input_int,u),1)
    

    def pred_ovl(self, input_int,model_list,subi_list,subj_list,unii_list,unij_list):
        '''
        model1: NN over the subdomain of smaller values
        model2: NN over teh subdomain of larger values
        sub1: smaller
        sub2: larger
        '''
        assert(len(model_list)==len(subi_list)==len(subj_list)==len(unii_list)==len(unij_list))
        input_int.requires_grad = True

        u_sum = 0
        for (model,subi,subj,unii,unij) in zip(model_list,subi_list,subj_list,unii_list,unij_list):

            u_sum += unnormalize2d( model(self.normalize2d(input_int,subi,subj)))*window2d(input_int,unii,unij)

   
        w_max =  torch.Tensor(self.w_list).max()

        u = (1/w_max)*torch.sin(w_max*input_int[:,1]).reshape(-1,1)+torch.tanh(w_max*input_int[:,0]).reshape(-1,1)*u_sum
        
  
        
        return torch.cat((input_int,u),1)
    ################################################################################################
    def predict(self):
        pred_list = []
        sol_list = [] 
        x_list = []

        for i in range(self.n_sub[0]): 
            for j in range(self.n_sub[1]):
                data_cent = self.pred_center(self.nn_list[i][j],self.cen_list[i][j],self.area[0]['sub'][i],self.area[1]['sub'][j])
                pred_list.append(data_cent[:,2])
                sol_list.append(self.exact_solution(data_cent[:,:2]).reshape(-1,))
                x_list.append(data_cent[:,:2])
                if j<=self.n_sub[1]-2:
                    data_wid  = self.pred_ovl(input_int = self.wid_list[i][j],
                                                model_list=[self.nn_list[i][j],self.nn_list[i][j+1]],
                                                subi_list=[self.area[0]['sub'][i],self.area[0]['sub'][i]],
                                                subj_list=[self.area[1]['sub'][j],self.area[1]['sub'][j+1]],
                                                unii_list=[self.area[0]['uni'][i],self.area[0]['uni'][i]],
                                                unij_list=[self.area[1]['uni'][j],self.area[1]['uni'][j+1]])
                    pred_list.append(data_wid[:,2])
                    sol_list.append(self.exact_solution(data_wid[:,:2]).reshape(-1,))
                    x_list.append(data_wid[:,:2])

                # ovl_2_len and ovl_4
                if i <= self.n_sub[0]-2:

                    data_len = self.pred_ovl(input_int = self.len_list[i][j],
                                                model_list=[self.nn_list[i][j],self.nn_list[i+1][j]],
                                                subi_list=[self.area[0]['sub'][i],self.area[0]['sub'][i+1]],
                                                subj_list=[self.area[1]['sub'][j],self.area[1]['sub'][j]],
                                                unii_list=[self.area[0]['uni'][i],self.area[0]['uni'][i+1]],
                                                unij_list=[self.area[1]['uni'][j],self.area[1]['uni'][j]])
                    pred_list.append(data_len[:,2])
                    sol_list.append(self.exact_solution(data_len[:,:2]).reshape(-1,))
                    x_list.append(data_len[:,:2])

                    if j<=self.n_sub[1]-2:
                        #oder is 1,2
                        #        3,4
                        data_4  = self.pred_ovl(input_int = self.four_list[i][j],
                                                model_list = [self.nn_list[i][j],self.nn_list[i][j+1],self.nn_list[i+1][j],self.nn_list[i+1][j+1]],
                                                subi_list  = [self.area[0]['sub'][i],self.area[0]['sub'][i],self.area[0]['sub'][i+1],self.area[0]['sub'][i+1]],
                                                subj_list  = [self.area[1]['sub'][j],self.area[1]['sub'][j+1],self.area[1]['sub'][j],self.area[1]['sub'][j+1]],
                                                unii_list  = [self.area[0]['uni'][i],self.area[0]['uni'][i],self.area[0]['uni'][i+1],self.area[0]['uni'][i+1]],
                                                unij_list  = [self.area[1]['uni'][j],self.area[1]['uni'][j+1],self.area[1]['uni'][j],self.area[1]['uni'][j+1]])
                        pred_list.append(data_4[:,2])
                        sol_list.append(self.exact_solution(data_4[:,:2]).reshape(-1,))
                        x_list.append(data_4[:,:2])
        return torch.concat(pred_list,0).detach(),torch.concat(sol_list,0).detach(), torch.concat(x_list,0).detach()
    
    #debug functions
    def plot_exact_points(self):
         
        plt.figure(figsize=(30,4.8))
        plt.grid(True, which="both", ls=":")
        for i in range(self.n_sub):
            x_cent = self.cen_set[i]
            y_cent = self.exact_solution(x_cent)
            plt.scatter(x_cent.detach(), y_cent.detach(), label="cent_{}".format(i),lw=2)
            if i <= (self.n_sub -2):
                x_ovl =  self.ovl_set[i]
                y_ovl = self.exact_solution(x_ovl)
                plt.scatter(x_ovl.detach(),y_ovl.detach(), label = "ovl_{}".format(i),lw=2)

        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend(ncol=6)
        plt.savefig('check_points.png')
    
    def plot_subarea(self):
        plt.figure(figsize=(15,15))
        plt.grid(True, which="both", ls=":")
      # Plot the input training points

        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]):
                if (i == 0) and (j==0):
                    plt.scatter(self.cen_list[i][j][:, 1].detach().numpy(), self.cen_list[i][j][:, 0].detach().numpy(),c='b', label="Cent points")
                else:
                    plt.scatter(self.cen_list[i][j][:, 1].detach().numpy(), self.cen_list[i][j][:, 0].detach().numpy(),c='b')

        
        # wid shu
        for i in range(self.n_sub[0]):
            for j in range(self.n_sub[1]-1):
                if (i == 0) and (j==0):
                    plt.scatter(self.wid_list[i][j][:, 1].detach().numpy(), self.wid_list[i][j][:, 0].detach().numpy(),c='g', label="Width points")
                else:
                    plt.scatter(self.wid_list[i][j][:, 1].detach().numpy(), self.wid_list[i][j][:, 0].detach().numpy(),c='g')
        
        # len heng
        for i in range(self.n_sub[0]-1):
            for j in range(self.n_sub[1]):
                if (i == 0) and (j==0):
                    plt.scatter(self.len_list[i][j][:, 1].detach().numpy(), self.len_list[i][j][:, 0].detach().numpy(),c='y', label="Length points")
                else:
                    plt.scatter(self.len_list[i][j][:, 1].detach().numpy(), self.len_list[i][j][:, 0].detach().numpy(),c='y')
       

        for i in range(self.n_sub[0]-1):
            for j in range(self.n_sub[1]-1):
                if (i == 0) and (j==0):
                    plt.scatter(self.four_list[i][j][:, 1].detach().numpy(), self.four_list[i][j][:, 0].detach().numpy(),c='r', label="Four overlapped points")
                else:
                    plt.scatter(self.four_list[i][j][:, 1].detach().numpy(), self.four_list[i][j][:, 0].detach().numpy(),c='r')
          

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.savefig('sub_area.png')

    def plotting(self,hist):


        output,exact_output,inputs = self.predict()


        fig, axs = plt.subplots(1, 3, figsize=(20, 4.8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=exact_output.detach(), cmap="jet",s=1)
        axs[0].set_xlabel("x1")
        axs[0].set_ylabel("x2")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output.detach(), cmap="jet",s=1)
        axs[1].set_xlabel("x1")
        axs[1].set_ylabel("x2")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Exact solution")
        axs[1].set_title("FBPINNs prediction")

        im3 = axs[2].plot(np.arange(1, len(hist) + 1), hist, label="FBPINNs")
        axs[2].set_xlabel("Traning Steps")
        axs[2].set_ylabel("L1 error")
        axs[2].set_title("Test error")
     
        axs[2].grid(True, which="both", ls=":")


        plt.savefig('results.png')

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")


if __name__ == '__main__':
    
    train = False
    n_int = 900*900
    n_layers = 2
    n_neurons = 16
    ovl=0.6
    fbpinn = FBPinns2D(n_int_=n_int,w_list=[15],n_layers=n_layers,n_neurons=n_neurons,n_subdomain=(15,15),overlap_len=ovl)
    
    n_epochs = 41000


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

        para_adam = []
        for i in range(fbpinn.n_sub[0]):
            for j in range(fbpinn.n_sub[1]):
                para_adam.append({'params':fbpinn.nn_list[i][j].parameters()})
        optimizer_ADAM = optim.Adam(para_adam,lr=float(0.001))
        start  =time.time()
        hist ,hist_error = fbpinn.fit(num_epochs=n_epochs,
                        optimizer=optimizer_ADAM,
                        verbose=True)
        end = time.time()
        print('The training time is {} s'.format(end-start))
        net = [[0 for i in range(fbpinn.n_sub[1])] for j in range(fbpinn.n_sub[0])]
        for i in range(fbpinn.n_sub[0]):
            for j in range(fbpinn.n_sub[1]):
                net[i][j] =  fbpinn.nn_list[i][j].state_dict()
        torch.save(net,'net_list_final.pth')
        torch.save(hist,"train_hist_final.pth")
        torch.save(hist_error,'hist_error_final.pth')
        # # plot the training loss
        # plt.figure(dpi=150)
        # plt.grid(True, which="both", ls=":")
        # plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        # plt.xscale("log")
        # plt.legend()
        # plt.savefig('trainloss.png')

        # #L1 errora
        # plt.figure(dpi=150)
        # plt.grid(True, which="both", ls=":")
        # plt.plot(np.arange(1, len(hist_error) + 1), hist_error, label="L1 error")
        # plt.xscale("log")
        # plt.legend()
        # plt.savefig('L1_error.png')
        
        #pred the training predictions
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
        nets = torch.load('./net_list_final.pth')
        hist = torch.load('./hist_error_final.pth')
        for i in range(15):
            for j in range(15):
                fbpinn.nn_list[i][j].load_state_dict(nets[i][j])
        fbpinn.plotting(hist=hist)
