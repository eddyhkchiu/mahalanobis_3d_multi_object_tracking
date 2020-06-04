# Metrics and matching algorithms for tracking

import os.path, copy, numpy as np, time, sys
from .utils.geometry_utils import diff_orientation_correction, convert_3dbox_to_8corner, iou3d
import json

from sklearn.utils.linear_assignment_ import linear_assignment

def pre_threshold_match(distance_matrix):
  to_max_mask = distance_matrix > mahalanobis_threshold
  distance_matrix[to_max_mask] = cfg.TRACKER.MATCH_THRESHOLD
  matched_indices = linear_assignment(distance_matrix)      # hungarian algorithm
  return matched_indices

def hungarian_match(distance_matrix):
  return linear_assignment(distance_matrix)                 # hungarian algorithm

def greedy_match(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      tracking_id_matches_to_detection_id[tracking_id] = detection_id
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices

def mahalanobis_metric(detections, trackers, **kwargs):
  """
    Creates matrix of mahalanobis distances between detections and tracks

    detections:  N x 7 
    trackers:    M x 8
    kwargs: {
      trks_S: N x 7 x 7
    }
    Returns score matrix [M x N]
    """
  score_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
  trks_S = kwargs['trks_S']
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
      diff = np.expand_dims(det - trk, axis=1) # 7 x 1
      # manual reversed angle by 180 when diff > 90 or < -90 degree
      corrected_angle_diff = diff_orientation_correction(det[3], trk[3])
      diff[3] = corrected_angle_diff
      score_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
  
  return score_matrix
  
def iou_metric(detections, trackers, **kwargs):
  """
    Creates matrix of negative IOU score between detections and tracks

    detections:  N x 7 
    trackers:    M x 8

    Returns score matrix [M x N]
    """
  score_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      det_8corner = convert_3dbox_to_8corner(det, True)
      trk_8corner = convert_3dbox_to_8corner(trk, True)
      score_matrix[d,t] = -iou3d(det_8corner,trk_8corner)[0]   
  
  return score_matrix
