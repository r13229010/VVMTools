from vvm_analysis import newVVMTools
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
testtools=newVVMTools('/data/mlcloud/r13229010/VVM/DATA/case_HW5a')
func_config = {"domain_range": (None, None, None, None, 64, 128)}
TKE_land = testtools.func_time_parallel(func=testtools.cal_TKE,time_steps=list(range(0, 721, 1)), func_config=func_config, cores=10)
plt.pcolormesh(TKE_land)
plt.show()