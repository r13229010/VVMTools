from vvm_analysis import newVVMTools
from vvmtools.plot import DataPlotter
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

testtools=newVVMTools('/data/mlcloud/r13229010/VVM/DATA/case_HW5a')
func_config = {"domain_range": (None, None, None, None, None, None)}
TKE_land = testtools.func_time_parallel(func=testtools.cal_wth,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
th = testtools.get_var_parallel('th',time_steps=list(range(0, 721, 1)),domain_range=func_config['domain_range'],compute_mean=True, axis=(1,2))
TKE_real = testtools.func_time_parallel(func=testtools.cal_TKE,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
Enstrophy_real = testtools.func_time_parallel(func=testtools.cal_Enstrophy,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
wth_bar = testtools.func_time_parallel(func=testtools.cal_wth,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
t05 = testtools.find_BL_boundary(th, "th_plus05K")
dthdz = testtools.find_BL_boundary(th, "dthdz")
tke= testtools.find_BL_boundary(TKE_real, "threshold",threshold=0.1)
Enstrophy = testtools.find_BL_boundary(Enstrophy_real, "threshold",threshold=0.1)
wth_pn, wth_min, wth_np, indice_02 = testtools.find_BL_boundary(wth_bar, "wth",threshold=1e-2)
print("t05.shape: " + str(t05.shape))
boundary_dict = {                   
    "θ + 0.5K": t05,                
    "max dθ/dz": dthdz,            
    "TKE": tke,
    "Enstrophy": Enstrophy,      
    "top(w′θ′ +)": wth_pn,         
    "min(w′θ′ )": wth_min,         
    "top(w′θ′ -)": wth_np         
}

# prepare expname and data coordinate
nx = 128; x = np.arange(nx)*0.2
ny = 128; y = np.arange(ny)*0.2
nz = 50;  z = np.arange(nz)*0.04
nt = 721; t = np.arange(nt)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00')
time = np.arange(nt)


expname  = 'HW7a'
figpath           = './fig/'
data_domain       = {'x':x, 'y':y, 'z':z, 't':t}
data_domain_units = {'x':'km', 'y':'km', 'z':'m', 't':'LocalTime'}
plotter = DataPlotter(exp=expname, figpath=figpath, domain=data_domain ,
         units=data_domain_units )
#for key, value in boundary_dict.items():
#    print(f"{key} size:", len(value))
wth_bar = np.hstack((np.full((nt,1),np.nan), wth_bar))
plotter.draw_zt(data = wth_bar.T, \
                      levels=np.arange(-0.04,0.041,0.005), \
                      extend='both', \
                      pblh_dicts=boundary_dict,\
                      cmap_name='bwr',\
                      title_left = 'vertical θ transport', \
                      title_right = 'PE whole domain', \
                      xlim = None, \
                      ylim = None,\
                      figname='HW7.1.png',\
               )

