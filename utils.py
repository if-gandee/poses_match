import numpy as np



def _find_transform_similarity(pts_1, pts_2):
    ''' find a similarity transformation that maps pts_1 to pts_2, which means
    pts_2 = pts_1.dot(R)*s+t.
    
    The method is described in:
    S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," 
    in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376-380, Apr 1991.
    
    return transmat, 4x4 matrix for 3d points
    '''
    pts_1 = np.atleast_2d(pts_1)
    pts_2 = np.atleast_2d(pts_2)
    
    ndim = pts_1.shape[-1]
    c1 = pts_1.mean(axis=0)
    p1 = pts_1 - c1
    var1= (p1**2).sum(axis=1).mean()
    
    c2 = pts_2.mean(axis=0)
    p2 = pts_2 - c2
    var2 = (p2**2).sum(axis=1).mean()
    
    #sigmat = 0
    #for i in xrange(len(p1)):
        #sigmat += np.outer(p2[i], p1[i])
    #sigmat /= len(p1)
    sigmat = np.einsum('ik,jk->ij', p2.T, p1.T)/len(p1)
    U,D,Vt = np.linalg.svd(sigmat)
    D = np.diag(D)
    
    S = np.eye(sigmat.shape[0])
    sigrank = np.linalg.matrix_rank(sigmat)
    if sigrank == ndim:
        #full rank
        if np.linalg.det(sigmat)<0:
            S[-1,-1] = -1
    elif sigrank == ndim-1:
        dval = np.linalg.det(U) * np.linalg.det(Vt.T)
        if dval < 0:
            S[-1,-1] = -1
    else:
        assert False, 'rank is less than ndim-1, similarity transformation cannot be found'
    
    # create left-multiply transformation matrix
    rotmat = U.dot(S).dot(Vt)
    scale = 1/var1 * np.trace(D.dot(S))
    tranvec = c2[:,np.newaxis] - scale * rotmat.dot(c1[:,np.newaxis])
    transmat = np.eye(4)
    transmat[:3,:3] = scale * rotmat
    transmat[:3,-1] = tranvec.flatten()
    
    # convert to right multiply format
    return transmat.T


def find_transform_similarity(pts_1, pts_2):
    ''' find a similarity transformation that maps pts_1 to pts_2, which means
    pts_2 = pts_1.dot(R)*s+t.
    
    The method is described in:
    S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," 
    in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376-380, Apr 1991.
    
    return transmat, 4x4 matrix for 3d points and 3x3 for 2d points
    '''
    pts_1 = np.atleast_2d(pts_1)
    pts_2 = np.atleast_2d(pts_2)
    
    ndim = pts_1.shape[-1]    
    npts = len(pts_1)
    
    if ndim == 2:
        pts_1 = np.column_stack((pts_1, np.zeros(npts)))
        pts_2 = np.column_stack((pts_2, np.zeros(npts)))
        tmat = _find_transform_similarity(pts_1,pts_2)
        output = np.eye(3)
        output[:2,:2] = tmat[:2,:2]
        output[-1,:2] = tmat[-1,:2]
    else:
        output = _find_transform_similarity(pts_1, pts_2)
    return output