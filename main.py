# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function

import os.path, copy, numpy as np, time, sys
import json

from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.detection.data_classes import DetectionBox 

from .utils.generic_utils import load_list_from_folder, fileparts, mkdir_if_missing
from .utils.geometry_utils import poly_area, box3d_vol, convex_hull_intersection, polygon_clip, \
        iou3d, roty, rotz, convert_3dbox_to_8corner, angle_in_range, diff_orientation_correction
from .utils.config import cfg, cfg_from_yaml_file, log_config_to_file
from . import algorithms, covariance

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info, 
                track_score=None, 
                tracking_name='car', 
                use_angular_velocity=False):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    if not use_angular_velocity:
      self.kf = KalmanFilter(dim_x=10, dim_z=7)       
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,0,0,1],
                            [0,0,0,1,0,0,0,0,0,0],  
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]])     
    
      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0]])
    else:
      # with angular velocity
      # [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
      self.kf = KalmanFilter(dim_x=11, dim_z=7)       
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0,0,0,1,0],
                            [0,0,0,1,0,0,0,0,0,0,1],  
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,1]])     
     
      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0]])

    # Initialize the covariance matrix, see covariance.py for more details
    cov = getattr(covariance, cfg.TRACKER.COVARIANCE)(tracking_name=tracking_name)
    self.kf.P = cov.P
    self.kf.Q = cov.Q
    self.kf.R = cov.R

    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info
    self.track_score = track_score
    self.tracking_name = tracking_name
    self.use_angular_velocity = use_angular_velocity

  def update(self, bbox3D, info): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x.reshape((-1, ))

class AB3DMOT(object):
  def __init__(self, max_age=2, min_hits=3, tracking_name='car'):
    """              
    observation: 
      [x, y, z, rot_y, l, w, h]
    state:
       [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.tracking_name = tracking_name
    self.use_angular_velocity = cfg.TRACKER.USE_ANGULAR_VELOCITY
    # Replace with dependency injection 
    self.match_threshold = cfg.TRACKER.MATCH_THRESHOLD
    self.score_metric_fn = getattr(algorithms,cfg.TRACKER.SCORE_METRIC)
    self.match_algorithm_fn = getattr(algorithms,cfg.TRACKER.MATCH_ALGORITHM)

  def update(self, dets_all, print_debug = False):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    #print('dets.shape: ', dets.shape)
    #print('info.shape: ', info.shape)
    #dets = dets[:, self.reorder]

    self.frame_count += 1

    if print_debug:
      for trk_tmp in self.trackers:
        print('trk_tmp.id: ', trk_tmp.id)

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    if print_debug:
      for trk_tmp in self.trackers:
        print('trk_tmp.id: ', trk_tmp.id)
    
    trks_S = []
    if cfg.TRACKER.SCORE_METRIC == 'mahalanobis_metric':
      trks_S = np.array([np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers])
        
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, print_debug=print_debug, trks_S=trks_S)
   
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0], info[d, :][0])
        detection_score = info[d, :][0][-1]
        trk.track_score = detection_score

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        detection_score = info[i][-1]
        track_score = detection_score
        trk = KalmanBoxTracker(dets[i,:], info[i, :], track_score, self.tracking_name, self.use_angular_velocity) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()  
        #d = d[self.reorder_back]

        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(np.array([d, trk.id+1, trk.track_score])) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return ret      # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,15 + 7))      

  def associate_detections_to_trackers(self, detections, trackers, print_debug=False, **kwargs):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 7 
    trackers:    M x 8
    kwargs: {
      trks_S: N x 7 x 
      ...
    }
    
    Returns 3 lists of matches [M, 2], unmatched_detections [?] and unmatched_trackers [?]
    """
    
    use_mahalanobis = cfg.TRACKER.SCORE_METRIC == 'mahalanobis_metric'
    
    if(len(trackers)==0):
      return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
    # score matrix between detections and trackers. Lower is better. Either Mahalonobis distance or - IOU
    score_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    if use_mahalanobis and print_debug:
      print('dets.shape: ', detections.shape)
      print('dets: ', detections)
      print('trks.shape: ', trackers.shape)
      print('trks: ', trackers)
      trks_S = kwargs['trks_S']
      print('trks_S.shape: ', trks_S.shape)
      print('trks_S: ', trks_S)
      S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
      S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7
      print('S_inv_diag: ', S_inv_diag)

    score_matrix = self.score_metric_fn(detections, trackers, **kwargs)
    matched_indices = self.match_algorithm_fn(score_matrix)

    if print_debug:
      print('score_matrix.shape: ', score_matrix.shape)
      print('score_matrix: ', score_matrix)
      print('matched_indices: ', matched_indices)

    unmatched_detections = []
    for d,det in enumerate(detections):
      if(d not in matched_indices[:,0]):
        unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
      if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
        unmatched_trackers.append(t)

    #filter out matched with high score (bad)
    matches = []
    for m in matched_indices:
      match = score_matrix[m[0],m[1]] < self.match_threshold
      if not match:
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
      else:
        matches.append(m.reshape(1,2))
    if(len(matches)==0):
      matches = np.empty((0,2),dtype=int)
    else:
      matches = np.concatenate(matches,axis=0)

    if print_debug:
      print('matches: ', matches)
      print('unmatched_detections: ', unmatched_detections)
      print('unmatched_trackers: ', unmatched_trackers)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def format_sample_result(sample_token, tracking_name, tracker):
  '''
  Input:
    tracker: (3):[[x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot(, rot_y_dot)], tracking_id, tracking_score]
  Output:
  sample_result {
    "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
    "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
    "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
    "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                       Note that the tracking_name cannot change throughout a track.
    "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                       We average over frame level scores to compute the track level score.
                                       The score is used to determine positive and negative tracks via thresholding.
  }
  '''
  box = tracker[0]
  rotation = Quaternion(axis=[0, 0, 1], angle=box[3]).elements
  sample_result = {
    'sample_token': sample_token,
    'translation': [box[0], box[1], box[2]],
    'size': [box[4], box[5], box[6]],
    'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
    'velocity': [box[7], box[8]],
    'tracking_id': str(int(tracker[1])),
    'tracking_name': tracking_name,
    'tracking_score': tracker[2]
  }

  return sample_result

def track_nuscenes(save_root):
  '''
  submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_map":      <bool>  -- Whether this submission uses map data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
  }
  
  '''
  save_dir = os.path.join(save_root, os.path.splitext(os.path.basename(cfg.DATASET.DETECTION_FILE))[0]); mkdir_if_missing(save_dir)
  output_path = os.path.join(save_dir, 'results_' + cfg.DATASET.DATA_SPLIT + '_probabilistic_tracking.json')

  nusc = NuScenes(version=cfg.DATASET.VERSION, dataroot=cfg.DATASET.DATA_ROOT, verbose=True)

  results = {}

  total_time = 0.0
  total_frames = 0

  with open(cfg.DATASET.DETECTION_FILE) as f:
    data = json.load(f)
  assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
    'See https://www.nuscenes.org/object-detection for more information.'

  all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
  meta = data['meta']
  print('meta: ', meta)
  print("Loaded results from {}. Found detections for {} samples."
    .format(cfg.DATASET.DETECTION_FILE, len(all_results.sample_tokens)))

  processed_scene_tokens = set()
  for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
    sample_token = all_results.sample_tokens[sample_token_idx]
    scene_token = nusc.get('sample', sample_token)['scene_token']
    if scene_token in processed_scene_tokens:
      continue
    first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
    current_sample_token = first_sample_token

    mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name) for tracking_name in cfg.TRACKER.CLASSES}

    while current_sample_token != '':
      results[current_sample_token] = []
      dets = {tracking_name: [] for tracking_name in cfg.TRACKER.CLASSES}
      info = {tracking_name: [] for tracking_name in cfg.TRACKER.CLASSES}
      for box in all_results.boxes[current_sample_token]:
        if box.detection_name not in cfg.TRACKER.CLASSES:
          continue
        q = Quaternion(box.rotation)
        angle = q.angle if q.axis[2] > 0 else -q.angle
        #print('box.rotation,  angle, axis: ', box.rotation, q.angle, q.axis)
        #print('box.rotation,  angle, axis: ', q.angle, q.axis)
        #[x, y, z, rot_y, l, w, h]
        detection = np.array([
          box.translation[0],  box.translation[1], box.translation[2],
          angle,
          box.size[0], box.size[1], box.size[2]])
        #print('detection: ', detection)
        information = np.array([box.detection_score])
        dets[box.detection_name].append(detection)
        info[box.detection_name].append(information)
        
      dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])}
        for tracking_name in cfg.TRACKER.CLASSES}

      total_frames += 1
      start_time = time.time()
      for tracking_name in cfg.TRACKER.CLASSES:
        if dets_all[tracking_name]['dets'].shape[0] > 0:
          trackers = mot_trackers[tracking_name].update(dets_all[tracking_name])
          # (N, 3)
          # [[x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot], tracking_id, tracking_score]
          # print('trackers: ', trackers)
          for i in range(len(trackers)):
            sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
            results[current_sample_token].append(sample_result)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      # get next frame and continue the while loop
      current_sample_token = nusc.get('sample', current_sample_token)['next']

    # left while loop and mark this scene as processed
    processed_scene_tokens.add(scene_token)

  # finished tracking all scenes, write output data
  output_data = {'meta': meta, 'results': results}
  with open(output_path, 'w') as outfile:
    json.dump(output_data, outfile, indent=2)

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))


if __name__ == '__main__':
  if len(sys.argv)!=3:
    print("Usage: python main.py cfg_file save_root")
    sys.exit(1)
  
  cfg_file = sys.argv[1]
  save_root = os.path.join(sys.argv[2], os.path.splitext(os.path.basename(cfg_file))[0])
  cfg_from_yaml_file(cfg_file, cfg)
  
  track_nuscenes(save_root)