import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from Common import NeuralNet, MultiVariatePoly
import time
from helper import *
import matplotlib.pyplot as plt
import time

# All scheduled implementation of FBPinns over the problemset:
# du/dx = w1*cos(w1*x)+w2*cos(w2*x)

class FBPinns:
    def __init__(self, n_int_,w_list, n_layers,n_neurons,n_subdomain=30, overlap_len=0.3):
        self.n_int = n_int_
        self.n_sub = n_subdomain
        self.w_list = w_list
        self.n_layer =  n_layers
        self.n_neuron = n_neurons
        self.overlap_len =overlap_len
        self.device =  torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else "cpu") #



        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-2*np.pi, 2*np.pi]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        # self.lambda_u = 10

        # List of NNs to approximate the solution over the subdomains
        self.nn_list = self.get_nnlist()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.cen_set, self.ovl_set,self.ovl_area,self.cen_area,self.sub_area, self.uni_area = self.get_subdomain_list()
        

     ################################################################################################
    # Function to return a list of NNs over different subdomains
    def get_nnlist(self):
        nn_list = []
        for i in range(self.n_sub):
            nn_list.append(NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=self.n_layer,
                                              neurons=self.n_neuron,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42).to(self.device))
        return nn_list

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
        u = torch.zeros_like(inputs)
        for i in range (len(self.w_list)):
            u += torch.sin(self.w_list[i]*inputs)
        return u



    


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
    
    
    def get_subdomain_list(self):
        sub_len  = (self.domain_extrema[0][1]-self.domain_extrema[0][0])/self.n_sub
        unif_bnd = torch.linspace(self.domain_extrema[0][0],self.domain_extrema[0][1],steps=self.n_sub+1)
        bnd_plus = unif_bnd + self.overlap_len/2
        bnd_min = unif_bnd - self.overlap_len/2
        unif_area = torch.Tensor([[unif_bnd[i],unif_bnd[i+1]] for i in range(self.n_sub)])
        overlap_area = torch.Tensor([[bnd_min[i+1],bnd_plus[i+1]] for i in range(self.n_sub-1)]) # Tensor (n_sub-1) * 2
        center_area = torch.Tensor([[bnd_plus[i],bnd_min[i+1]] for i in range(self.n_sub)]) # Tensor n_sub * 2
        #The firs and last term of center_area should be confined in the domain
        center_area[0][0]  = self.domain_extrema[0][0]
        center_area[-1][1] = self.domain_extrema[0][1] 
        subdomain_area = torch.Tensor([[bnd_min[i],bnd_plus[i+1]] for i in range(self.n_sub)])
        #The firs and last term of subdomain area should be confined in the domain
        subdomain_area[0][0] = self.domain_extrema[0][0]
        subdomain_area[-1][1] = self.domain_extrema[0][1]
        
        input_int, output_int = self.add_interior_points()         
        center_list = []
        overlap_list = []
        for i,area in enumerate(overlap_area):
            ind = (input_int >= area[0]) * (input_int <= area[1])
            overlap_list.append(input_int[ind].reshape(-1,1).to(self.device))
           
        assert(len(overlap_list)==self.n_sub-1)

        for i,area in enumerate(center_area):
            ind = (input_int >= area[0]) * (input_int <= area[1])
            center_list.append(input_int[ind].reshape(-1,1).to(self.device))
  
        assert(len(center_list)==self.n_sub)


        return center_list,overlap_list,overlap_area,center_area,subdomain_area, unif_area
        
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
        sum = torch.zeros_like(input_int)
        for i in self.w_list:
            sum += i*torch.cos(i*input_int)
        return sum


    def get_center_residual(self, model, input_int, sub):
        '''
        subdomain determines the scale of the normalization for the input variable
        '''
        input_int.requires_grad = True
        u = unnormalize(model(normalize(input_int, subdomain=sub)))
        w_max =  torch.Tensor(self.w_list).max()
        u = torch.tanh(w_max*input_int)*u
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]

        residual = l2_loss(grad_u,self.f_x(input_int))
        with torch.no_grad():
         error = l1_loss(self.exact_solution(input_int),u)

        #residual = l2_loss(self.exact_solution(input_int),u)
        return residual,error
    

    def get_ovl_residual(self, input_int,model1,model2,sub1,sub2,uni1,uni2):
        '''
        model1: NN over the subdomain of smaller values
        model2: NN over teh subdomain of larger values
        sub1: smaller
        sub2: larger
        '''
        input_int.requires_grad = True

        u_1 = unnormalize( model1(normalize(input_int,subdomain=sub1)))*window(input_int,uni1,sigma=0.005)
        u_2 = unnormalize( model2(normalize(input_int,subdomain=sub2)))*window(input_int,uni2,sigma=0.005)
        w_max =  torch.Tensor(self.w_list).max()

        u = (u_1+u_2) * torch.tanh(w_max*input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]

        residual = l2_loss(grad_u,self.f_x(input_int))
        with torch.no_grad():
          error = l1_loss(self.exact_solution(input_int),u)
        
        return residual, error
    


    # Function to compute the mean of the physics loss of the center domains and the overlapped domains
    def compute_loss(self, verbose=True):
        r_cent_list = []
        r_ovl_list = []
        err_cent_list = []
        err_ovl_list = []
        for i in range(self.n_sub): 
            r_cent,err_cent = self.get_center_residual(self.nn_list[i],self.cen_set[i],self.sub_area[i])
            r_cent_list.append(r_cent)
            err_cent_list.append(err_cent)
            if i <= self.n_sub-2:
                r_ovl, err_ovl  = self.get_ovl_residual(input_int = self.ovl_set[i],
                                               model1 = self.nn_list[i],
                                               model2 = self.nn_list[i+1],
                                               sub1 = self.sub_area[i],
                                               sub2 = self.sub_area[i+1],
                                               uni1 = self.uni_area[i],
                                               uni2 = self.uni_area[i+1])
                r_ovl_list.append(r_ovl)
                err_ovl_list.append(err_ovl)
      

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

        print('Final Loss: ', history[-1])
        print('Final L1 error: ', hist_error[-1])

        return history, hist_error
    
    def pred_center(self, model, input_int, sub):
        '''
        subdomain determines the scale of the normalization for the input variable
        '''

        u = unnormalize(model(normalize(input_int, subdomain=sub)))
        w_max =  torch.Tensor(self.w_list).max()
        u = torch.tanh(w_max*input_int) * u


        return u
    
    def plot_window_funcitons(self,sig):

        plt.figure(figsize=(45,4.8))
        plt.grid(True, which="both", ls=":")
        plt.ylim((-0.5,1.5))
        for i in range(self.n_sub):
            ovl_set=[]
            if i == 0:

                ovl_set=torch.sort(torch.cat((self.cen_set[i],self.ovl_set[i]),0),0)[0]
            elif i == self.n_sub-1:
                ovl_set=torch.sort(torch.cat((self.cen_set[i],self.ovl_set[i-1]),0))[0]
            else:
                ovl_set=torch.sort(torch.cat((self.ovl_set[i-1],self.cen_set[i],self.ovl_set[i]),0))[0]
            
            y_wind = window(ovl_set,self.uni_area[i],sigma=sig)
            plt.scatter(ovl_set.detach(), y_wind.detach(), s=0.5)


        plt.xlabel("x")
        plt.ylabel("window funtion")
        plt.legend(ncol=6)
        plt.savefig('windowfunc_{}.png'.format(sig))

    def pred_ovl(self, input_int,model1,model2,sub1,sub2,uni1,uni2):
        '''
        model1: NN over the subdomain of smaller values
        model2: NN over teh subdomain of larger values
        sub1: smaller
        sub2: larger
        '''

        u_1 = window(input_int,uni1)*unnormalize( model1(normalize(input_int,subdomain=sub1)))
        u_2 = window(input_int,uni2)*unnormalize( model2(normalize(input_int,subdomain=sub2)))
        w_max =  torch.Tensor(self.w_list).max()
        u = (u_1+u_2) * torch.tanh(w_max*input_int)
        
        return u
    ################################################################################################
    def predict(self,input_int):
        #divide
        center_list = []
        overlap_list = []
        for i,area in enumerate(self.ovl_area):
            ind = (input_int >= area[0]) * (input_int <= area[1])
            overlap_list.append(input_int[ind].reshape(-1,1).to(self.device))
           
        assert(len(overlap_list)==self.n_sub-1)

        for i,area in enumerate(self.cen_area):
            ind = (input_int >= area[0]) * (input_int <= area[1])
            center_list.append(input_int[ind].reshape(-1,1).to(self.device))
  
        assert(len(center_list)==self.n_sub)

        center_points =  torch.concat(center_list,0)
        ovl_points = torch.concat(overlap_list,0)
        #predict
        y_cent_list = []
        y_ovl_list = []
        for i in range(self.n_sub): 
            y_cent = self.pred_center(self.nn_list[i],center_list[i],self.sub_area[i])
            y_cent_list.append(y_cent)
            if i <= self.n_sub-2:
                y_ovl  = self.pred_ovl(input_int = overlap_list[i],
                                               model1 = self.nn_list[i],
                                               model2 = self.nn_list[i+1],
                                               sub1 = self.sub_area[i],
                                               sub2 = self.sub_area[i+1],
                                               uni1 = self.uni_area[i],
                                               uni2 = self.uni_area[i+1])
                y_ovl_list.append(y_ovl)
        return torch.concat(y_cent_list,0).detach(),torch.concat(y_ovl_list,0).detach(), center_points.detach(),ovl_points.detach()
    
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
        plt.figure(figsize=(30,4.8))
        plt.grid(True, which="both", ls=":")
        for i in range(self.n_sub):

            area = self.sub_area[i]
            x = torch.linspace(area[0], area[1], 50).reshape(-1,1)
            if i%2 == 0:
                y_even = torch.ones_like(x)*0.2
                plt.plot(x, y_even.detach(), label="subdomain_{}".format(i),lw=4)
            if i%2 ==1:
                y_odd = torch.ones_like(x)*0
                plt.plot(x,y_odd, label = "subdomain_{}".format(i),lw=4)
        plt.xlabel("x")
        plt.ylim((-4,5))
        plt.legend(ncol=6)
        plt.savefig('subdomains.png')

    def plot_window_funcitons(self,sig,title):

        plt.figure(figsize=(30,4.8))
        plt.grid(True, which="both", ls=":")
        plt.ylim((-0.5,1.5))
        for i in range(self.n_sub):
            ovl_set=[]
            if i == 0:

                ovl_set=torch.sort(torch.cat((self.cen_set[i],self.ovl_set[i]),0),0)[0]
            elif i == self.n_sub-1:
                ovl_set=torch.sort(torch.cat((self.cen_set[i],self.ovl_set[i-1]),0))[0]
            else:
                ovl_set=torch.sort(torch.cat((self.ovl_set[i-1],self.cen_set[i],self.ovl_set[i]),0))[0]
            
            y_wind = window(ovl_set,self.uni_area[i],sigma=sig)
            plt.scatter(ovl_set.detach(), y_wind.detach(), s=0.5)
        
        plt.xlabel("x")
        plt.ylabel("window funtion")
        plt.title(title)
        plt.savefig('windowfunc_{}.png'.format(sig))


if __name__ == '__main__':
    work_dir='/home/xdjf/dlsc/1d/ex0.005/'
    train = True
    n_int = 64 * 300
    w = [2**i for i in range(1,6)]
    n_layers = 2
    n_neurons = 16
    ovl=0.12
    fbpinn = FBPinns(n_int_=n_int,w_list=w,n_layers=n_layers,n_neurons=n_neurons,n_subdomain=64,overlap_len=ovl)

    n_epochs = 15000
    para = []
    for i in fbpinn.nn_list:
        para += list(i.parameters())
       
    
    optimizer_LBFGS = optim.LBFGS(para,
                                lr=float(0.1),
                                max_iter=50000, 
                                max_eval=50000,
                                history_size=150,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)

    optimizer_ADAM = optim.Adam([{'params':net.parameters()} for net in fbpinn.nn_list],
                                    lr=float(0.0005))

    if train == True:
        start  =time.time()
        hist ,hist_error = fbpinn.fit(num_epochs=n_epochs,
                        optimizer=optimizer_ADAM,
                        verbose=True)
        end = time.time()
        print('The training time is {} s'.format(end-start))

        net = [net.state_dict() for net in fbpinn.nn_list]
        torch.save(net,'{}final_netlist_ovl{}_nint{}.pth'.format(work_dir,ovl,n_int))
        torch.save(hist,(work_dir+'training_loss.pth'))
        torch.save(hist_error,(work_dir+'Test_error.pth'))

        # plot the training loss
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.savefig(work_dir+'trainloss.png')

        #L1 error
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist_error) + 1), hist_error, label="L1 error")
        plt.xscale("log")
        plt.legend()
        plt.savefig(work_dir+'L1_error.png')
        
        #pred the training predictions
        x = torch.linspace(fbpinn.domain_extrema[0][0],fbpinn.domain_extrema[0][1], 2000).reshape(-1,1)
        y_exact = fbpinn.exact_solution(x)
        y_pred_cen,y_pred_ovl, x_cent,x_ovl  = fbpinn.predict(x)
        plt.figure(figsize=(30,4.8))
        plt.grid(True, which="both", ls=":")
 
        plt.scatter(x_cent, fbpinn.exact_solution(x_cent), label="Ground Truth",s=10)
        plt.scatter(x_cent, y_pred_cen, label="Network Prediction",s=10)
        plt.scatter(x_ovl, fbpinn.exact_solution(x_ovl), label="Ground Truth",s=10)
        plt.scatter(x_ovl, y_pred_ovl, label="Network Prediction",s=10)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.savefig(work_dir+'pred.png')
    else:
        #fbpinn.plot_exact_points()
        fbpinn.plot_window_funcitons(sig=0.005)
