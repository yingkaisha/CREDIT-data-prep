'''
This script is the same as PP_ERA5_plevel_upper_subset.py but for
derived variables such as specific total water Q = qv + ql + qw + qc

Yingkai Sha
ksha@ucar.edu
'''


import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

import multiprocessing
from dask.distributed import Client
from dask_jobqueue import PBSCluster

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# ==================================================================================== #
# subsetting function
def integral_conserved_subset_all_vars(ds, level_p, ind_select):
    # Precompute the level differences and midpoints
    diff_level_p = np.diff(level_p)
    
    # Create a helper function to compute the integral for each column
    def integral_conserved_subset_1d(x_column):
        x_column_midpoint = 0.5 * (x_column[1:] + x_column[:-1])
        x_column_area = x_column_midpoint * diff_level_p
        
        # Allocate the output array
        out_column_a = np.empty(len(ind_select)-1)
        out_column_a.fill(np.nan)
        
        for i_ind, ind in enumerate(ind_select[:-1]):
            ind_start = ind
            ind_end = ind_select[i_ind+1]
            out_column_a[i_ind] = np.sum(x_column_area[ind_start:ind_end]) / (level_p[ind_end] - level_p[ind_start])
        
        return out_column_a
    
    # Apply the function along the 'level' dimension and specify output_sizes in dask_gufunc_kwargs
    ds_out = xr.apply_ufunc(
        integral_conserved_subset_1d, ds,
        input_core_dims=[['level']],
        output_core_dims=[['new_level']],
        vectorize=True,  # Broadcast across other dimensions
        dask='parallelized',  # Enable Dask parallelism if ds is chunked
        dask_gufunc_kwargs={
            'allow_rechunk': True,  # Allow rechunking if necessary
            'output_sizes': {'new_level': len(ind_select)-1}  # Specify the size of the new dimension
        },
        output_dtypes=[float]
    )
    
    return ds_out

# ==================================================================================== #
# get year from input
year = int(args['year'])

# configs
config_name = os.path.realpath('../data_config_6h.yml')
with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ==================================================================================== #
# the sub-folder to store data
base_dir = conf['zarr_opt']['save_loc'] + 'upper_subset/' 
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# ==================================================================================== #
# target levels to subset
level_p_select = np.array(conf['zarr_opt']['subset_level'])
level_midpoints = 0.5 * (level_p_select[1:] + level_p_select[:-1])

# ==================================================================================== #
# zarr enconding options
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

chunk_size = conf['zarr_opt']['chunk_size_4d']

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d']['time'],
                             conf['zarr_opt']['chunk_size_4d']['level'],
                             conf['zarr_opt']['chunk_size_4d']['latitude'],
                             conf['zarr_opt']['chunk_size_4d']['longitude']))
dict_encoding = {}
dict_encoding['specific_total_water'] = {'compressor': compress, **chunk_size_4d}

# ==================================================================================== #
# main

load_name = conf['RDA']['save_loc'] + 'upper_air/' + conf['RDA']['prefix'] + '_Q_{}.zarr'.format(year)

# ------------------------------------------------------------------------------------ #
# check if Dask client exists and shut it down
if 'client' in locals() and isinstance(client, Client):
    client.shutdown()
    print('...shutdown existing Dask client...')
else:
    print('Dask client does not exist, bulilding ...')

# set up the Dask cluster
project_num = 'NCIS0010' #'NAML0001'

cluster = PBSCluster(
    account=project_num,
    walltime='12:00:00',
    cores=1,
    memory='70GB',
    shared_temp_directory='/glade/derecho/scratch/ksha/tmp/',
    queue='casper'
)
cluster.scale(jobs=50)
# cluster.adapt(minimum=10, maximum=80)

client = Client(cluster)

print(f"Cluster dashboard: {client.dashboard_link}")

# ------------------------------------------------------------------------------------ #
# subsetting operations
ds_plevel = xr.open_zarr(load_name, consolidated=True)

level_p = np.array(ds_plevel['level'])
mask = np.isin(level_p, level_p_select)
ind_select = np.where(mask)[0]

ds_plevel = ds_plevel.chunk({'level': -1})
ds_plevel_subset = integral_conserved_subset_all_vars(ds_plevel, level_p, ind_select)
ds_plevel_subset = ds_plevel_subset.assign_coords(new_level=level_midpoints)
ds_plevel_subset = ds_plevel_subset.rename({'new_level': 'level'})
ds_plevel_subset = ds_plevel_subset.transpose('time', 'level', 'latitude', 'longitude')
ds_plevel_subset = ds_plevel_subset.chunk(chunk_size)

save_name = base_dir + conf['zarr_opt']['prefix'] + '_Q_{}.zarr'.format(year)
ds_plevel_subset.to_zarr(save_name, mode="w", consolidated=True, compute=True, encoding=dict_encoding)

# ------------------------------------------------------------------------------------ #
# shutdown dask cluster and client
print('... shutting down Dask client and cluster ...')
cluster.scale(0)
client.close()
cluster.close()

# removing Dask worker files
print('...removing Dask worker files...')
fns_rm = sorted(glob('./dask-worker*'))
print(f"Found {len(fns_rm)} Dask worker files.")
for fn in fns_rm:
    if os.path.exists(fn):
        os.remove(fn)
        print(f"Removed: {fn}")
    else:
        print(f"File not found: {fn}")

print('...all done...')

