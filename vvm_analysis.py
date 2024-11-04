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
    
    def Enstrophy(self, t, func_config):
        zeta = np.squeeze(self.get_var("zeta", t, numpy=True,domain_range=func_config["domain_range"]))
        eta = np.squeeze(self.get_var("eta", t, numpy=True,domain_range=func_config["domain_range"]))
        xi = np.squeeze(self.get_var("xi", t, numpy=True,domain_range=func_config["domain_range"]))        
        if zeta.shape != eta:
            eta = np.squeeze(self.get_var("eta_2", t, numpy=True,domain_range=func_config["domain_range"]))
        zeta_inter = (zeta[:, :, 1:] + zeta[:, :, :-1] + zeta[:, 1:, :] + zeta[:, :-1, :])[1:, :, :] / 4
        eta_inter = (eta[:, :, 1:] + eta[:, :, :-1] + eta[1:, :, :] + eta[:-1, :, :])[:, 1:, :] / 4
        xi_inter = (xi[1:, :, :] + xi[:-1, :, :] + xi[:, 1:, :] + xi[:, :-1, :])[:, :, 1:] / 4
        return np.nanmean((eta_inter**2 + zeta_inter**2 + xi_inter**2),axis = (1,2))

testtools=newVVMTools('/data/mlcloud/r13229010/VVM/DATA/case_HW5a')