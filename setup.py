#########################################
#Setup configuration and global variables
########################################
import os
os.environ["NUMBA_PARFOR_MAX_TUPLE_SIZE"] = "110"

parallel=True
nojit=False
cache=False
woman = 1
man = 2