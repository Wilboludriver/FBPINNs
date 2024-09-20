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
import os
import argparse

#This module is the basical implementation of Pinns 

class Pinns:
    def __init__(self, n_int_,w_list, n_layers,n_neurons):
        self.n_int = n_int_

        self.w_list = w_list
        self.n_layer =  n_layers
        self.n_neuron = n_neurons

        self.device =  torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else "cpu") #



        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-2*np.pi, 2*np.pi]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        # self.lambda_u = 10


        # F Dense NN to approximate the solution of the underlying heat equation
        self.model = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=self.n_layer,
                                              neurons=self.n_neuron,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_int = self.assemble_datasets()

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


    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):

        input_int, output_int = self.add_interior_points()         # S_int
        # training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return  training_set_int



    def f_x(self,input_int):
        sum = torch.zeros_like(input_int)
        for i in self.w_list:
            sum += i*torch.cos(i*input_int)
        return sum


    def get_center_residual(self,  input_int):
        '''
        subdomain determines the scale of the normalization for the input variable
        '''
        input_int.requires_grad = True
        u = unnormalize(self.model(normalize(input_int, subdomain=self.domain_extrema[0])))
        w_max =  torch.Tensor(self.w_list).max()
        u = torch.tanh(w_max*input_int)*u
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]

        residual = l2_loss(grad_u,self.f_x(input_int))
        with torch.no_grad():
         error = l1_loss(self.exact_solution(input_int),u)

        #residual = l2_loss(self.exact_solution(input_int),u)
        return residual,error
    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    
    def predict(self,input_int):
        with torch.no_grad():
            u = unnormalize(self.model(normalize(input_int, subdomain=self.domain_extrema[0])))
            w_max =  torch.Tensor(self.w_list).max()
            u = torch.tanh(w_max*input_int)*u

        return u


    def compute_loss(self,inp_train_int, verbose=True):

        r_int,err_int = self.get_center_residual(inp_train_int)
     

        loss = torch.log10(torch.mean(r_int))
        err = torch.log10(torch.mean(err_int))
        if verbose: print("Total loss: ", round(loss.item(), 4), "| Fitting err: ", round(err.item(), 4))

        return loss,err

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        hist_err = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, (inp_train_int, u_train_int) in enumerate(self.training_set_int):
                def closure():
                    optimizer.zero_grad()
                    loss,err = self.compute_loss(inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    hist_err.append(loss.item())

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history, hist_err

    ################################################################################################
    def plotting(self):
        
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=exact_output.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")

        plt.show()

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pinns-training')

    parser.add_argument('--nlayer', default= 2, type=int,
                        help='number of layers')

    parser.add_argument('--nneuron', default= 16, type=int,
                        help='number of neurons')
    
    parser.add_argument('--nepoch', default= 10, type=int,
                        help='number of neurons')

    args = parser.parse_args()
    train = True
    n_int = 30 * 200
    n_layers = args.nlayer
    n_neurons = args.nneuron
    work_dir ='/home/xdjf/dlsc/1d/pinn_baseline/{}l_{}n/'.format(n_layers,n_neurons)
    folder = os.path.exists(work_dir)

    if not folder:               
        os.makedirs(work_dir) 
  
    pinn = Pinns(n_int_=n_int,w_list=[1,15],n_layers=n_layers,n_neurons=n_neurons)
    n_epochs = args.nepoch
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

    optimizer_ADAM = optim.Adam(pinn.model.parameters(),
                                    lr=float(0.0005))

    if train == True:
        start  =time.time()
        hist ,hist_error = pinn.fit(num_epochs=n_epochs,
                        optimizer=optimizer_ADAM,
                        verbose=True)
        end = time.time()
        
        print('The training time is {} s'.format(end-start))

        torch.save(pinn.model.state_dict(),'{}final_netlist__nint{}.pth'.format(work_dir,n_int))
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
        x = torch.linspace(pinn.domain_extrema[0][0],pinn.domain_extrema[0][1], 2000).reshape(-1,1)
        y_exact = pinn.exact_solution(x)
        y_pred = pinn.predict(x).detach().numpy()
        plt.figure(figsize=(30,4.8))
        plt.grid(True, which="both", ls=":")
 
        plt.scatter(x, y_exact, label="Ground Truth",s=10)
        plt.scatter(x, y_pred, label="Network Prediction",s=10)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.savefig(work_dir+'pred.png')
    else:
        pinn.plot_exact_points()

