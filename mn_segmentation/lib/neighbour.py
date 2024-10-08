import json
import os
import argparse

def assign_parent_nuc(nuc_coord, mn_coord):
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

def add_parents(data):
  for i in range(len(data)):
    nuc_coord = data[i]['nuclei']['coord']
    mn_coord = data[i]['micronuclei']['coord']
    ind = assign_parent_nuc(nuc_coord, mn_coord)
    data[i]['micronuclei']['parent'] = ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add parent nuclei to mn.')
    parser.add_argument('--src', required=True,
                        help='the json file with processed dataset information')

    args = parser.parse_args()
    data = json.load(open(args.src))
    add_parents(data)