import numpy as np
from utils.config import cfg

class BaselineCovariance(object):
  '''
  Define different Kalman Filter covariance matrix as in AB3DMOT
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
  def __init__(self, **kwargs):
    self.num_states = 10
    self.num_observations = 7
    self.P = np.eye(self.num_states)
    self.Q = np.eye(self.num_states)
    self.R = np.eye(self.num_observations)

    self.P[self.num_observations:, self.num_observations:] *= 1000.
    self.P *= 10.
    self.Q[self.num_observations:, self.num_observations:] *= 0.01


class KittiCovariance(object):
  '''
  Define different Kalman Filter covariance matrix based on Kitti data
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
  def __init__(self, **kwargs):
    
    self.num_states = 10
    self.num_observations = 7
    self.P = np.eye(self.num_states)
    self.Q = np.eye(self.num_states)
    self.R = np.eye(self.num_observations)

    # from kitti stats
    self.P[0,0] = 0.01969623
    self.P[1,1] = 0.01179107
    self.P[2,2] = 0.04189842
    self.P[3,3] = 0.52534431
    self.P[4,4] = 0.11816206
    self.P[5,5] = 0.00983173
    self.P[6,6] = 0.01602004
    self.P[7,7] = 0.01334779
    self.P[8,8] = 0.00389245 
    self.P[9,9] = 0.01837525

    self.Q[0,0] = 2.94827444e-03
    self.Q[1,1] = 2.18784125e-03
    self.Q[2,2] = 6.85044585e-03
    self.Q[3,3] = 1.10964054e-01
    self.Q[4,4] = 0
    self.Q[5,5] = 0
    self.Q[6,6] = 0
    self.Q[7,7] = 2.94827444e-03
    self.Q[8,8] = 2.18784125e-03
    self.Q[9,9] = 6.85044585e-03

    self.R[0,0] = 0.01969623
    self.R[1,1] = 0.01179107
    self.R[2,2] = 0.04189842
    self.R[3,3] = 0.52534431
    self.R[4,4] = 0.11816206
    self.R[5,5] = 0.00983173
    self.R[6,6] = 0.01602004
  
  
class NuScenesCovariance(object):
  '''
  Define different Kalman Filter covariance matrix based on NuScenes data
  This includes angular velocity
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
  def __init__(self, **kwargs):
    
    self.num_states = 10
    self.num_observations = 7
    self.P = np.eye(self.num_states)
    self.Q = np.eye(self.num_states)
    self.R = np.eye(self.num_observations)

    # nuscenes
    # see get_nuscenes_stats.py for the details on  how the numbers come from
    #Kalman Filter state: [x, y, z, rot_z, l, w, h, x_dot, y_dot, z_dot, rot_z_dot]

    P = {
      'bicycle':    [0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823, 0.01169572, 0.01295084, 0.04560422, 0.04097244, 0.01725477, 1.21635902],
      'bus':        [0.17546469, 0.13818929, 0.05947248, 0.1979503 , 0.78867322, 0.05507407, 0.06684149, 0.13263319, 0.11508148, 0.05033665, 0.22529652],
      'car':        [0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802, 0.02359175, 0.02455134, 0.08120681, 0.08224643, 0.02266425, 0.99492726],
      'motorcycle': [0.04052819, 0.0398904 , 0.01511711, 1.06442726, 0.03291016, 0.00957574, 0.0111605 , 0.0437039 , 0.04327734, 0.01465631, 1.30414345],
      'pedestrian': [0.03855275, 0.0377111 , 0.02482115, 2.0751833 , 0.02286483, 0.0136347 , 0.0203149 , 0.04237008, 0.04092393, 0.01482923, 2.0059979 ],
      'trailer':    [0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601, 0.06354783, 0.10500918, 0.2138643 , 0.19625241, 0.05231335, 0.97082174],
      'truck':      [0.14862173, 0.1444596 , 0.05417157, 0.73122169, 0.69387238, 0.05484365, 0.07748085, 0.10683797, 0.10248689, 0.0378078 , 0.76188901]
    }

    Q = {
      'bicycle':    [1.98881347e-02, 1.36552276e-02, 5.10175742e-03, 1.33430252e-01, 0, 0, 0, 1.98881347e-02, 1.36552276e-02, 5.10175742e-03, 1.33430252e-01],
      'bus':        [1.17729925e-01, 8.84659079e-02, 1.17616440e-02, 2.09050032e-01, 0, 0, 0, 1.17729925e-01, 8.84659079e-02, 1.17616440e-02, 2.09050032e-01],
      'car':        [1.58918523e-01, 1.24935318e-01, 5.35573165e-03, 9.22800791e-02, 0, 0, 0, 1.58918523e-01, 1.24935318e-01, 5.35573165e-03, 9.22800791e-02],
      'motorcycle': [3.23647590e-02, 3.86650974e-02, 5.47421635e-03, 2.34967407e-01, 0, 0, 0, 3.23647590e-02, 3.86650974e-02, 5.47421635e-03, 2.34967407e-01],
      'pedestrian': [3.34814566e-02, 2.47354921e-02, 5.94592529e-03, 4.24962535e-01, 0, 0, 0, 3.34814566e-02, 2.47354921e-02, 5.94592529e-03, 4.24962535e-01],
      'trailer':    [4.19985099e-02, 3.68661552e-02, 1.19415050e-02, 5.63166240e-02, 0, 0, 0, 4.19985099e-02, 3.68661552e-02, 1.19415050e-02, 5.63166240e-02],
      'truck':      [9.45275998e-02, 9.45620374e-02, 8.38061721e-03, 1.41680460e-01, 0, 0, 0, 9.45275998e-02, 9.45620374e-02, 8.38061721e-03, 1.41680460e-01]
    }

    R = {
      'bicycle':    [0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823, 0.01169572, 0.01295084],
      'bus':        [0.17546469, 0.13818929, 0.05947248, 0.1979503 , 0.78867322, 0.05507407, 0.06684149],
      'car':        [0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802, 0.02359175, 0.02455134],
      'motorcycle': [0.04052819, 0.0398904 , 0.01511711, 1.06442726, 0.03291016, 0.00957574, 0.0111605 ],
      'pedestrian': [0.03855275, 0.0377111 , 0.02482115, 2.0751833 , 0.02286483, 0.0136347 , 0.0203149 ],
      'trailer':    [0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601, 0.06354783, 0.10500918],
      'truck':      [0.14862173, 0.1444596 , 0.05417157, 0.73122169, 0.69387238, 0.05484365, 0.07748085]
    }
    
    self.P = np.diag(P[kwargs['tracking_name']])
    self.Q = np.diag(Q[kwargs['tracking_name']])
    self.R = np.diag(R[kwargs['tracking_name']])
    
    if not cfg.TRACKER.USE_ANGULAR_VELOCITY:
        self.P = self.P[:-1,:-1]
        self.Q = self.Q[:-1,:-1]
