"""
module for functions that assign mn to a parent nuc
"""

import json
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mn_segmentation.lib.image_encode import rle_to_mask, mask2rle

def extract_polygons(mask):
    """Extracts polygons from a binary mask using OpenCV findContours"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cnt.reshape(-1, 2) for cnt in contours]  # Convert contours to list of points
    return polygons

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Compute the shortest distance from a point (px, py) to a line segment (x1, y1) -> (x2, y2)"""
    A = np.array([x1, y1])
    B = np.array([x2, y2])
    P = np.array([px, py])
    
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB) if np.dot(AB, AB) > 0 else 0
    
    if t < 0:
        closest = A
    elif t > 1:
        closest = B
    else:
        closest = A + t * AB
    
    return np.linalg.norm(P - closest)

def closest_polygon_edge_distance(point, mask):
    """Find the closest polygon to a given point using contours from a binary mask"""
    px, py = point
    polygons = extract_polygons(mask)
    
    min_distance = float('inf')
    closest_polygon = None

    for polygon in polygons:
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1) % len(polygon)]  # Loop back to first point
            dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
            
            if dist < min_distance:
                min_distance = dist
                closest_polygon = polygon

    return closest_polygon, min_distance

def assign_parent_nuc_edge(nuc, mn_coord):
  '''
  assign parent nuc to mn

  Parameter:
    nuc_pos: list of coords [[x,y],...]

  Return:
    ind: index of parent nuc for each mn
  '''
  ind = []
  for x1,y1 in mn_coord:
    min_dist = float('inf')
    min_ind = -1
    for i, c in enumerate(nuc["coord"]):
      x2,y2 = c
      dist = (x1-x2)**2 + (y1-y2)**2
      if dist < min_dist:
        min_dist = dist
        min_ind = i

    min_edg = float('inf')
    id = -1
    for i, rle in enumerate(nuc["mask"]):
      x2,y2 = nuc["coord"][i]
      dist = (x1-x2)**2 + (y1-y2)**2
      if dist - min_dist < 300:
        mask = rle_to_mask(rle,nuc["height"],nuc["width"])
        _, distance = closest_polygon_edge_distance([x1,y1], mask)
        if distance < min_edg:
           min_edg = distance
           id = i
    ind.append(id)

  return ind

def assign_parent_nuc_center(nuc_coord, mn_coord):
  '''
  assign parent nuc to mn

  Parameter:
    nuc_pos: list of coords [[x,y],...]

  Return:
    ind: index of parent nuc for each mn
  '''
  ind = []
  for x1,y1 in mn_coord:
    min_dist = float('inf')
    min_ind = -1
    for i, c in enumerate(nuc_coord):
      x2,y2 = c
      dist = (x1-x2)**2 + (y1-y2)**2
      if dist < min_dist:
        min_dist = dist
        min_ind = i
    ind.append(min_ind)
  return ind

def add_parents(data, mod):
  if mod == "center":
    for i in range(len(data)):
      nuc_coord = data[i]['nuclei']['coord']
      mn_coord = data[i]['micronuclei']['coord']
      ind = assign_parent_nuc_center(nuc_coord, mn_coord)
      data[i]['micronuclei']['parent'] = ind
  else:
    for i in range(len(data)):
      mn_coord = data[i]['micronuclei']['coord']
      ind = assign_parent_nuc_edge(data[i]['nuclei'], mn_coord)
      data[i]['micronuclei']['parent'] = ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add parent nuclei to mn.')
    parser.add_argument('--src', required=True,
                        help='the json file with processed dataset information')
    parser.add_argument('--mod', required=True,
                        help='the mod to assign parent')

    args = parser.parse_args()
    data = json.load(open(args.src))
    mod = args.mod
    add_parents(data, mod)
