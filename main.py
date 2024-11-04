from vvm_analysis import newVVMTools
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
testtools=newVVMTools('/data/mlcloud/r13229010/VVM/DATA/case_HW5a')
func_config = {"domain_range": (None, None, None, None, 64, 128)}
TKE_land = testtools.func_time_parallel(func=testtools.cal_wth,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
th = testtools.get_var_parallel('th',time_steps=list(range(0, 721, 1)),domain_range=func_config['domain_range'],compute_mean=True, axis=(1,2))
t05 = testtools.find_BL_boundary(th, "th_plus05K")
plt.pcolormesh(np.arange(721), testtools.DIM["zc"][1:], TKE_land.T)
plt.plot(t05)
plt.show()