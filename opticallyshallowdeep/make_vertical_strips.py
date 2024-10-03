import numpy as np
def make_vertical_strips(full_img):
    '''use to save ram, process bigger images faster, and it overlaps so middle image is not 
    distorted from how edge pixels are handled'''
    
    # number of dimensions 
    n_dim = full_img.ndim
    
    if n_dim == 2: 
        height, width = full_img.shape #this is done so strips do not have artifacts from kernals
        overlap_size = 16 #size of overlap, max tile size is 15, so there is a 1px buffer
        strip1 = full_img[:, :width//5 + overlap_size]#left overlap
        strip2 = full_img[:, width//5: 2*width//5+ overlap_size]# left half overlap
        strip3 = full_img[:, 2*width//5:3*width//5 + overlap_size]#left overlap
        strip4 = full_img[:, 3*width//5:4*width//5+ overlap_size]#left overlap
        strip5 = full_img[:, 4*width//5:width]# no overlap
    elif n_dim == 3: 
        height, width, _ = full_img.shape #this is done so strips do not have artifacts from kernals
        overlap_size = 16 #size of overlap, max tile size is 15, so there is a 1px buffer
        strip1 = full_img[:, :width//5 + overlap_size, :]#left overlap
        strip2 = full_img[:, width//5: 2*width//5+ overlap_size, :]# left half overlap
        strip3 = full_img[:, 2*width//5:3*width//5 + overlap_size, :]#left overlap
        strip4 = full_img[:, 3*width//5:4*width//5+ overlap_size, :]#left overlap
        strip5 = full_img[:, 4*width//5:width, :]# no overlap       
    else:
        import sys
        sys.exit('Unknown dimension(s) of input imagery to be splited into strips')
          
    return [strip1,strip2,strip3,strip4,strip5]