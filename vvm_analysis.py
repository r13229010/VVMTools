import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime
import vvmtools



class newVVMTools(vvmtools.analyze.DataRetriever):
    def __init__(self, case_path):
        super().__init__(case_path)
        

    def cal_TKE(self, t, func_config):
        u = np.squeeze(self.get_var("u", t, numpy=True,domain_range=func_config["domain_range"]))
        v = np.squeeze(self.get_var("v", t, numpy=True,domain_range=func_config["domain_range"]))
        w = np.squeeze(self.get_var("w", t, numpy=True,domain_range=func_config["domain_range"]))
        u_inter = (u[:, :, 1:] + u[:, :, :-1])[1:, 1:] / 2
        v_inter = (v[:, 1:] + v[:, :-1])[1:, :, 1:] / 2
        w_inter = (w[1:] + w[:-1])[:, 1:, 1:] / 2
        TKE = np.mean(u_inter ** 2 + v_inter ** 2 + w_inter ** 2, axis=(1, 2))
        return TKE
    
    def cal_Enstrophy(self, t, func_config):
        zeta = np.squeeze(self.get_var("zeta", t, numpy=True,domain_range=func_config["domain_range"]))
        eta = np.squeeze(self.get_var("eta", t, numpy=True,domain_range=func_config["domain_range"]))
        xi = np.squeeze(self.get_var("xi", t, numpy=True,domain_range=func_config["domain_range"]))   
        if zeta.shape != eta.shape:
            eta = np.squeeze(self.get_var("eta_2", t, numpy=True,domain_range=func_config["domain_range"]))
        # zeta_inter = (zeta[:, :, 1:] + zeta[:, :, :-1] + zeta[:, 1:, :] + zeta[:, :-1, :])[1:, :, :] / 4
        # eta_inter = (eta[:, :, 1:] + eta[:, :, :-1] + eta[1:, :, :] + eta[:-1, :, :])[:, 1:, :] / 4
        # xi_inter = (xi[1:, :, :] + xi[:-1, :, :] + xi[:, 1:, :] + xi[:, :-1, :])[:, :, 1:] / 4
        
        xi_inter = (xi[:, 1:, 1:] + xi[:, :-1, 1:] + xi[:, 1:, :-1] + xi[:, :-1, :-1])[1:] / 4
        eta_inter = (eta[1:, :, 1:] + eta[:-1, :, 1:] + eta[1:, :, :-1] + eta[:-1, :, :-1])[:, 1:] / 4
        zeta_inter = (zeta[1:, 1:] + zeta[:-1, 1:] + zeta[1:, :-1] + zeta[:-1, :-1])[:, :, 1:] / 4
 
        E =  np.mean(eta_inter ** 2 + zeta_inter ** 2 + xi_inter ** 2, axis=(1, 2))
        return E

    def cal_wth(self, t, func_config):
        w = np.squeeze(self.get_var("w", t, numpy=True,domain_range=func_config["domain_range"]))
        th = np.squeeze(self.get_var("th", t, numpy=True,domain_range=func_config["domain_range"]))[1:]
        w_inter = (w[1:] + w[:-1]) / 2
        wbar = np.squeeze(self.get_var("w", t, numpy=True,compute_mean=True,axis=(1,2)))
        wbar_inter = (wbar[1:] + wbar[:-1]) / 2
        thbar = np.squeeze(self.get_var("th", t, numpy=True,compute_mean=True,axis=(1,2)))[1:]
        wprime = w_inter - wbar_inter.reshape(len(wbar_inter),1,1)
        thprime = th - thbar.reshape(len(thbar),1,1)
        wthbar = np.mean(wprime * thprime,axis=(1,2))
        return wthbar

    def find_BL_boundary(self, var, howToSearch, threshold=0.01):
        zc = self.DIM["zc"]/1000
        zc_expand = np.meshgrid(zc,var[:,0],indexing='ij')[0]
        if howToSearch == "th_plus05K":
            index = np.argmin(np.abs(var - (var[:,0].reshape(len(var[:,0]),1) + 0.5)),axis=1)
            print(index)
            height = zc[index]
            
            return height


        elif howToSearch == "dthdz":
            var_gradient = np.gradient(var, zc, axis=1)
            max_gradient_indices = np.argmax(var_gradient, axis=1)
            max_gradient_heights = zc[max_gradient_indices]
            print('shape of max gradient heights: ',np.shape(max_gradient_heights) )
            return max_gradient_heights

        elif howToSearch == "threshold":
            boundary_heights = np.zeros(len(var[:,0]))
            for timestep in range(len(var[:,0])):
                if np.max(np.abs(var[timestep])) >= threshold:
                    boundary_heights[timestep] = zc[np.argmin(np.abs(var[timestep]-threshold))]
                else:
                    boundary_heights[timestep] = np.nan
            return boundary_heights

        elif howToSearch == "wth":
            boundary_pn = np.zeros(len(var[:,0]))
            boundary_min = np.zeros(len(var[:,0]))
            boundary_np = np.zeros(len(var[:,0]))
            indice_03 = np.zeros(len(var[:,0]))
            for timestep in range(len(var[:,0])):
                if np.max(np.abs(var[timestep])) <= threshold:
                    boundary_pn[timestep] = 0
                    boundary_min[timestep] = 0
                    boundary_np[timestep] = 0
                elif np.max(var[timestep, np.argmax((var[timestep])):]) < threshold:
                    print(np.max(var[timestep, np.argmax((var[timestep])):]))
                    boundary_min[timestep] = zc[np.argmin(var[timestep])]
                    boundary_pn[timestep] = zc[np.argmin(np.abs(var[timestep, 1:np.argmin(var[timestep])]))]
                    boundary_np[timestep] = 0#np.nan
                else:
                    boundary_min[timestep] = zc[np.argmin(var[timestep])]
                    boundary_pn[timestep] = zc[np.argmin(np.abs(var[timestep, 1:np.argmin(var[timestep])]))]
                    indice_01 = np.argmin(var[timestep]) # minimum
                    indice_02 = indice_01 + np.argmax((var[timestep, indice_01:])) # second large
                    boundary_np[timestep] = zc[indice_01 + np.argmin(np.abs(var[timestep, indice_01:indice_02] + threshold))]
                    indice_03[timestep] = (zc[indice_02])
            return boundary_pn, boundary_min, boundary_np, indice_03

        else:
            print("Without this searching approach")
