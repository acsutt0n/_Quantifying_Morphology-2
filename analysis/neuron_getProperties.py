# retrieve simple properties from a geo instance

from neuron_readExportedGeometry import *
import numpy as np
import math, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches



# helper functions
def name(geo):
  return geo.fileName.split('/')[-1].split('.')[0]


def farthest_pt(pts):
  dmax = 0
  for i in pts:
    for j in pts:
      if dist3(i,j) > dmax:
        dmax = dist3(i,j)
  return dmax


def farthest_pt2(pts):
  sumpts = [sum(p) for p in pts]
  minpt = pts.index(sumpts.index(min(sumpts)))
  maxpt = pts.index(sumpts.index(max(sumpts)))
  dmax = dist3(minpt, maxpt)
  return dmax*1.5
    


def checko(obj):
  unique_files, unique_items, unique_cells = None, None, None
  if type(obj) is not dict:
    print('Only works for dictionaries'); return 
  if len(obj['files']) != len(np.unique(obj['files'])):
    print('Duplicates found in files!')
  unique_files = len(np.unique(obj['files']))
  for k in obj.keys():
    if k != 'files' and k != 'cellTypes' and k != 'cellType':
      if len(obj[k]) != len(np.unique(obj[k])):
        print('Duplicates found in %s!' %k)
      unique_items = len(np.unique(obj[k]))
  try:
    unique_cells = len(np.unique(obj['cellTypes']))
  except:
    unique_cells = len(np.unique(obj['cellType']))
  print('Contents: %i unique files, %i unique items, %i cell types'
         %(unique_files, unique_items, unique_cells))
  return


def nodex(n):
  return [n.x, n.y, n.z]


def node3(n0, n1):
  return dist3(nodex(n0),nodex(n1))


def union(list1, list2, rall=False):
  """
  Returns the common segment between two lists. Union assumes there is only
  one common segment. First, it assumes the segment is not included in
  the lists, so it crawls through the neighbors of elements of both lists. If
  it finds more than 1 common segment, it assumes the segment is included
  in one of the lists, and then returns the common segment found by neighbors.
  Rall - returns list of common segments. 
  """
  def union1(list1, list2): 
    for a in list1:
      for n in a.neighbors:
        if n in list2:
          return a
    return None
  def union2(list1, list2, rall=False):
    u = []
    for a in list1:
      for b in list2:
        for n in a.neighbors:
          if n in b.neighbors:
            if n not in u:
              u.append(n)
    if rall is True and len(u) != 1:
      # print('Found %i matches' %len(u))
      return u
    if len(u) > 1:
      return union1(list1, list2)
    elif len(u) == 1:
      return u[0]
    else:
      return None
  return union2(list1, list2, rall)



#######################################################################
# prepare for analysis -- load up hoc files
# default cellTypes (spreadsheet):
cellTypes =  ['LG', 'LG', 'LG', 'LG', 'LP', 'LP', 'LP', 'LP', 'PD', \
              'PD', 'PD', 'PD', 'GM', 'PD']

def get_hocfiles(directory='/home/alex/data/morphology/morphology-hoc-files/morphology/analyze/'):
  fils = os.listdir(directory)
  hocfiles = [i for i in fils if i.split('.')[-1]=='hoc']
  hocfiles = [directory+i for i in hocfiles]
  return hocfiles
  

def get_geofiles(directory='/home/alex/data/morphology/morphology-hoc-files/morphology/analyze/'):
  hocfiles = get_hocfiles(directory)
  geofiles = [demoReadsilent(g) for g in hocfiles]
  return geofiles, hocfiles
  





#######################################################################
# branch angles

def dist3(pt0, pt1):
  if len(pt0) == len(pt1) and len(pt0) == 3:
    return math.sqrt(sum([(pt0[i]-pt1[i])**2 for i in range(3)]))
  else:
    print('dimension mismatch')
    print(pt0, pt1)




def get_angle(pt0, midpt, pt1):
  if pt0 in [midpt, pt1] or pt1 in [midpt, pt0] or midpt in [pt0,pt1]:
    print('Some points are the same!')
    print(pt0, midpt, pt1)
  PT0 = dist3(pt1, midpt)
  PT1 = dist3(pt0, midpt)
  MIDPT = dist3(pt0, pt1)
  try:
    ang = math.acos( (MIDPT**2 - PT1**2 - PT0**2) / (2*PT1*PT0) )
    ang = ang*180/math.pi
  except:
    ang = 'nan'
  return ang



def find_points(seg0, seg1):
  seg0list, seg1list = [], []
  pt0where, pt1where, midwhere = None, None, None
  switchdict = {0: -1, -1: 0}
  # make a list of the node locations
  for n in seg0.nodes:
    seg0list.append([n.x,n.y,n.z])
  for n in seg1.nodes:
    seg1list.append([n.x,n.y,n.z])
    # find the common node, then use that to find the distinct ones
  for n in seg0list:
    if n in seg1list:
      midpt = n
  if seg0list.index(midpt) != 0:
    pt0where = 0
    pt0 = seg0list[0]
  else:
    pt0where = -1
    pt0 = seg0list[-1]
  if seg1list.index(midpt) != 0:
    pt1where = 0
    pt1 = seg1list[0]
  else:
    pt1where = -1
    pt1 = seg1list[-1]
  
  f = True
  if pt0 == pt1 or pt0==midpt:
    f = False
    if pt0where == 0:
      try:
        pt0=seg0list[1]
        f = True
      except:
        pass
    elif pt0where == -1:
      try:
        pt0=seglist[-2]
        f = True
      except:
        pass
  if pt0 == pt1 or pt1==midpt:
    if pt1where == 0:
      try:
        pt1=seg1list[1]
        f = True
      except:
        pass
    elif pt1where == -1:
      try:
        pt1=seg1list[-2]
        f = True
      except:
        pass
  if f == False:
    print('Tried to find new coordinates, but failed. Skipping')
  if pt0 in [midpt, pt1] or pt1 in [midpt, pt0] or midpt in [pt0,pt1]:
    print(seg0list, seg1list)
  #print('pt0 at %i, pt1 at %i' %(pt0where, pt1where))
  if pt1 and pt0 and midpt:
    return pt0, midpt, pt1
  else:
    print('could not figure out segments %s and %s' %(seg0.name, seg1.name))
    print(seg0list, seg1list)
    return [False]


def branch_angles(geo, outfile=None):
  angles = []
  for b in geo.branches:
    for n in b.neighbors:
      pts = find_points(n, b)
      if len(pts) == 3:
        pt0, midpt, pt1 = pts[0], pts[1], pts[2]
      angles.append(get_angle(pt0, midpt, pt1))
  angles = [a for a in angles if a!='nan']
  if outfile is not None:
    with open('temp_angles.txt', 'w') as fOut:
      for a in angles:
        fOut.write('%.10f, \n' %a)
  return angles

"""
def get_subtrees(geo):
  # Generate a main path, subtrees come off the main path
  main = []
  def add_main(seg):
    maxn, candidate = 0, None
    for n in seg.neighbors:
      if len(n.neighbors) > maxn:
        maxn = len(n.neighbors)


def ordered_angles(geo):
"""


#######################################################################
# path length and tortuosity

def path_lengths(geo):
  tips, tipinds = geo.getTipIndices()
  pDF = PathDistanceFinder(geo, geo.soma, 0.5)
  tipsegs = [geo.segments[i] for i in tips]
  path = [pDF.distanceTo(x,y) for x, y in zip(tipsegs, tipinds)]
  tort = [pDF.tortuosityTo(x,y) for x, y in zip(tipsegs, tipinds)]
  return path, tort


def path_lengths2(geo):
  # if FilamentIndex != geo.segments[index], use this: 
  tips, tipinds = geo.getTipIndices()
  tipsegs = [i for i in geo.segments if geo.getFilamentIndex(i) in tips]
  pDF = PathDistanceFinder(geo, geo.soma, 0.5)
  path, tort = [], []
  for x, y in zip(tipsegs, tipinds):          
    try:
      p, t = pDF.distanceTo(x,y), pDF.tortuosityTo(x,y)
      path.append(p)
      #~ tort.append(t)
    except:
      continue
  return path, tort



def wen_tortuosity(geolist):
  """
  Returns pairs = [Euclidean, PathLengths] list, should be ~ 1, and 
  normalized tortuosity index (T) ~ normalized path length (l), 
  should be T = l/R -1. Inspired by figure 4E&F from Wen...Chklovskii, 2009
  """
  import scipy.stats as stats
  def seg_int(name):
    try:
      beh = int(name.split('[')[1].split(']')[0])
      return beh
    except:
      try:
        beh = int(name.split('_')[-1])
        return beh
      except:
        print(name)
    return 
  edists, plengths = [], []
  for g in geolist:
    tipsegs, tiplocs = g.getTipIndices()
    pDF = PathDistanceFinder(g, g.soma)
    tempedists, tempplengths = [], []
    for seg in g.segments:
      if seg_int(seg.name) in tipsegs:
        tempedists.append(dist3(g.soma.coordAt(0.5), seg.coordAt(0.5)))
        tempplengths.append(pDF.distanceTo(seg, 0.5))
    edists.append(np.mean(tempedists))
    plengths.append(np.mean(tempplengths))
  return edists, plengths
  
    
  
  




#######################################################################
# sholl stuff

def interpoint_dist(geo):
  # NOT CURRENTLY USED
  # determine the distances between successive points
  def nodex(node):
    return [node.x, node.y, node.z]
  dists = []
  for s in geo.segments:
    for n in range(len(s.nodes)-1):
      dists.append(dist3(nodex(s.nodes[n]), nodex(s.nodes[n+1])))
  print('Mean distance (%i points): %.5f +/- %.5f' 
         %(len(dists), np.mean(dists), np.std(dists)))
  return dists


def can_branch_be_added_again(geo, branch, sholl, key):
  # NOT CURRENTLY USED
  # profile branch
  soma = geo.soma.nodeAt(0)
  dD = [np.sign(node3(branch.nodes[n+1],soma)-node3(branch.nodes[n],soma))
        for n in range(len(branch.nodes)-1)]
  dLoc = [node3(i,soma) for i in branch.nodes[:-1]]
  changepts = []
  last_i = dD[0]
  for i in dD:
    if np.sign(i) != np.sign(last_i):
      changepts.append([dLoc[dD.index(i)], np.sign(i)])
      last_i = np.sign(i) # if there's a change, update the i
  # now figure out how many can be added
  if len(changepts) == 0:
    return False
  # if it only loops back once and hasn't been added twice, add it again
  elif len(changepts) == 1:
    if changepts[0][1] == -1: # if loops back
      if float(key) > changepts[0][0]:
        if sholl[key][1].count(branch) <= 1: # make 
          return True
    elif changepts[0][1] == 1: # if loops OUT
      if float(key) > changepts[0][0]:
        if sholl[key][1].count(branch) <= 1:
          return True
  else: # multiple change pts
    if sholl[key][1].count(branch) >= len(changepts) + 1:
      return False # already used all its changepts
    else:
      # assume forward order
      c = sholl[key][1].count(branch)
      change_sholls = [i[0] for i in changepts]
      if float(key) > min(change_sholls) and float(key) < max(change_sholls):
        return
  return


def interpolate_nodes(geo, return_nodes=False):
  # NOT CURRENTLY USED
  # find the most common distance betwixt successive nodes and then,
  # when successive nodes leave integer multiples of this distance
  # interpolate the difference to 'even' it out
  def nodex(node):
    return [node.x, node.y, node.z]
    
  def interp(pt1, pt2, ints):
    Xs = np.linspace(pt1[0], pt2[0], ints)
    Ys = np.linspace(pt1[1], pt2[1], ints)
    Zs = np.linspace(pt1[2], pt2[2], ints)
    return [[Xs[i],Ys[i],Zs[i]] for i in range(len(Xs))]
    
  dist = np.median(interpoint_dist(geo))
  pts = []
  segcount = -1
  for s in geo.segments:
    segcount = segcount + 1
    if segcount % 100 == 0:
      print('Completed %i/%i segments ' 
             %(segcount,len(geo.segments)))
    for n in range(len(s.nodes)-1):
      # if too far between nodes, interpolate
      if dist3(nodex(s.nodes[n]),nodex(s.nodes[n+1])) > 2 * dist:
        integer_interpolate = int((dist3(nodex(s.nodes[n]),
                                         nodex(s.nodes[n+1])))
                                   /dist)
        new_pts = interp(nodex(s.nodes[n]),nodex(s.nodes[n+1]),
                         integer_interpolate)
      # else just add the regular node pts
      else:
        new_pts = [nodex(s.nodes[n]), nodex(s.nodes[n+1])]
      # add the points as long as they don't already exist in pts
      for p in new_pts:
        if p not in pts:
          pts.append(p)
  # now should have all the points
  soma = geo.soma.coordAt(0.5)
  distances = []
  for p in pts:
    distances.append(dist3(soma, p))
  if return_nodes:
    return distances, pts
  else:
    return distances


def hooser_sholl(geo, sholl_lines=1000):
  """
  Sholl analysis without repeats or missed counts. Sholl_lines can be
  integer, whereupon the program creates that many evenly-spaced radii,
  or can be a vector whose values are used for sholl radii.
  """
  def get_neighbors(geo, branch, neb_segs):
    # For each possible neighbor, if it shares more than 1 node in common
    # with the branch, don't add it (it's probably the branch itself)
    possible_nodes = branch.nodes
    possible_segs, possible_locs = [], []
    for s in neb_segs:
      count = 0
      for n in s.nodes:
        if n in possible_nodes:
          count = count + 1
          if s.nodes.index(n) == len(s.nodes):
            possible_locs.append(-1)
          else:
            possible_locs.append(s.nodes.index(n))
          possible_segs.append(s)
      if count > 1:
        for j in range(count):
          possible_segs.pop()
          possible_locs.pop()
    # get the actual location of
    next_locs = []
    for i in possible_locs:
      if i == 0:
        next_locs.append(1)
      elif i == -1:
        next_locs.append(-2)
    possible_segs, possible_locs = possible_segs[0], possible_locs[0]
    return possible_segs.nodes[possible_locs]
    
  # helper functions
  def cross_node(geo, branch, nodeNum, sholl):
    # Determine if a sholl line is crossed by this node->next node
    soma = geo.soma.nodeAt(0)
    nebs = None
    #try: # if there is a next node, add it
    next_dist = node3(branch.nodes[nodeNum+1], soma)
    #except: # if not, find a ('the'?) neighboring node
    #  print('error!')
    #  try:
    #    nebs = [i[2] for i in branch.neighborLocations if i[0]==1]
    #  except:
    #    nebs = None
    # if this is the last node and there and neighbors with this node,
    # get the next node from the neighbors
    #if nebs:
    #  next_dist = get_neighbors(geo, branch, nebs)
    #else:
    #  return sholl
    node_dist = node3(branch.nodes[nodeNum], soma)
    for i in [float(k) for k in sholl.keys()]:
      if i < max([next_dist, node_dist]) and \
        i > min([next_dist, node_dist]):
          sholl[str(i)][0] = sholl[str(i)][0] + 1
          sholl[str(i)][1].append(branch)
    return sholl
      
  soma = geo.soma.nodeAt(0)
  sholl = {} # dictionary of crossings as keys, numcrossings as [0] (int)
             # and branches that cross (as objects? names?) as [1] (list)
  # integer mode
  if type(sholl_lines) is int:
    dists = [node3(n,soma) for n in geo.nodes]
    dists.sort()
    d99 = dists[int(len(dists)*.99)] # get 99% of the nodes
    lines = np.linspace(min(dists), d99, sholl_lines)
    for l in lines:
      sholl[str(l)] = [0,[]]
  # list mode
  elif type(sholl_lines) is list or type(sholl_lines) is np.ndarray:
    for l in sholl_lines:
      sholl[str(l)] = [0, []]
  else:
    print('sholl_lines input must be int or list or ndarray!')
    print('instead got %s' %str(type(sholl_list)))
    return None
  # go through branches and nodes and tabulate crossings
  for b in geo.branches:
    for nodeNum in range(len(b.nodes)-1): # here was change -1
      sholl = cross_node(geo, b, nodeNum, sholl)
  sholl_keys = list(sholl.keys())
  float_keys = [float(i) for i in sholl_keys]
  float_keys.sort()
  sholl_count = [sholl[str(i)][0] for i in float_keys]
  
  return [float_keys, sholl_count], sholl



######################################################################    
# sub-tree analysis


def sholl_color(geo, outfile, interdist=1.):
  # This color-codes hoc points based on sholl distances.
  pDF = PathDistanceFinder(geo, geo.soma)
  tips, tipPositions = geo.getTips()
  paths, _ = path_lengths2(geo)
  maxp = max(paths)
  nodes, dists = [], []
  for s in geo.segments:
    if s.length > interdist:
      x0, y0, z0 = s.coordAt(0)[0],s.coordAt(0)[1], s.coordAt(0)[2]
      x1, y1, z1 = s.coordAt(1)[0],s.coordAt(1)[1], s.coordAt(1)[2]
      temp_nodes = [np.linspace(x0, x1, int(s.length/interdist)),
                    np.linspace(y0, y1, int(s.length/interdist)),
                    np.linspace(z0, z1, int(s.length/interdist))]
      #print(np.shape(temp_nodes))
      for t in range(np.shape(temp_nodes)[1]):
        nodes.append([temp_nodes[0][t],temp_nodes[1][t],temp_nodes[2][t]])
        dists.append(pDF.distanceTo(s, 0.5)/maxp)
    else: # Add at least one node per segment
      nodes.append(s.coordAt(0.5))
      dists.append(pDF.distanceTo(s, 0.5)/maxp)
  if len(dists) != len(nodes):
    print('Warning! No. nodes: %i, No. distances: %i' %(len(nodes), len(nodes)))
  with open(outfile, 'w') as fOut:
    for i in range(len(nodes)):
      fOut.write('%.5f %.5f %.5f %.5f' %(nodes[i][0], nodes[i][1], 
                                         nodes[i][2], dists[i]))
      fOut.write('\n')
  print('%s file written' %outfile)
  return
  

def axons_endpoints(geo, outfile=None, Format='matlab'):
  # This prints possible axons and their start and end points. Feed
  # the output into shollAxons.m to select only the "true" axon(s).
  axons = []
  for s in geo.branches: # Operates on BRANCHES!
    if "Axon" in s.tags or "axon" in s.tags:
      axons.append([geo.branches.index(s), s.coordAt(0), s.coordAt(1)])
  if outfile is not None:
    print('Found %i potential branch axons' %len(axons))
    with open(outfile, 'w') as fOut:
      for a in axons:
        fOut.write('%i %.5f %.5f %.5f\n%i %.5f %.5f %.5f' 
                   %(a[0], a[1][0], a[1][1], a[1][2],
                     a[0], a[2][0], a[2][1], a[2][2]))
    return
  else:
    for a in axons:
      if Format == 'matlab':
        print('%i %.5f %.5f %.5f\n%i %.5f %.5f %.5f' 
              %(a[0], a[1][0], a[1][1], a[1][2],
                a[0], a[2][0], a[2][1], a[2][2]))
      else:
        print(a[0], a[1], a[2])
    print('Found %i potential branch axons' %len(axons))
    return a



def axon_help(geo, x=None, y=None, z=None):
  for s in geo.segments:
    for g in [0,1]:
      if x is not None:
        if int(s.coordAt(g)[0]) == x:
          if y is not None:
            if int(s.coordAt(g)[1]) == y:
              print(geo.segments.index(s), s.coordAt(g))
          else:
            print(geo.segments.index(s), s.coordAt(g))
  return



def simple_axon(geo, axon, thing='branch'):
  # Turn the axon into a NeuronGeometry.Segment (but NOT a branch!)
  def mid_pt(geo):
    midPt = [np.mean([s.coordAt(0)[0] for s in geo.segments]),
             np.mean([s.coordAt(0)[1] for s in geo.segments]),
             np.mean([s.coordAt(0)[2] for s in geo.segments])]
    return midPt
  #
  def distal_seg(geo, axon, midPt): # If axon==branch, return the most distal segment
    dists = [dist3([n.x,n.y,n.z], midPt) for n in axon.nodes]
    node = [x for (y,x) in sorted(zip(dists, axon.nodes))][-1]
    for s in geo.segments:
      if node in s.nodes:
        return s
  #
  midPt = mid_pt(geo)
  if thing == 'branch':
    return distal_seg(geo, geo.branches[axon], midPt)
  elif thing == 'seg' or thing == 'segment':
    return geo.segments[axon]



def axon_path(geo, axons=None, things=None, outfile=None, interdist=1.):
  # Given a geofile and an axon (default axon if none provided), this
  # returns the subtrees between the soma and that axon.
  #
  print(axons, things)
  if axons is not None and things is not None:
    if type(axons) is not list:
      axons = [axons]
    if things is not None:
      if type(things) is not list:
        things = [things]
    if len(axons) != len(things):
      if len(things) == 1:
        things = [things[0] for t in axons]
      else:
        print('Things should be len(axons) or length 1 (if all are same type)')
        return
    beh = [simple_axon(geo, a, t) for a, t in zip(axons, things)]
  elif axons is not None and things is None:
    beh = [a for a in axons]
  else:
    print(axons, things)
    return
  #
  pDF = PathDistanceFinder(geo, geo.soma)
  print(beh)
  paths = [pDF.pathTo(b) for b in beh]
  # If no outfile, just return the path
  if outfile is None:
    return paths
  else: # Else, write it to a file (with interpolation as before)
    nodes, dists = [], []
    for path in paths:
      for p in path:
        if p.length > interdist:
          x0, y0, z0 = p.coordAt(0)[0],p.coordAt(0)[1], p.coordAt(0)[2]
          x1, y1, z1 = p.coordAt(1)[0],p.coordAt(1)[1], p.coordAt(1)[2]
          temp_nodes = [np.linspace(x0, x1, int(p.length/interdist)),
                        np.linspace(y0, y1, int(p.length/interdist)),
                        np.linspace(z0, z1, int(p.length/interdist))]
          #print(np.shape(temp_nodes))
          for t in range(np.shape(temp_nodes)[1]):
            nodes.append([temp_nodes[0][t],temp_nodes[1][t],temp_nodes[2][t]])
            #dists.append(pDF.distanceTo(s, 0.5)/maxp)
        else: # Add at least one node per segment
          nodes.append(p.coordAt(0.5))
          #dists.append(pDF.distanceTo(s, 0.5)/maxp)
    with open(outfile, 'w') as fOut:
      for i in range(len(nodes)):
        fOut.write('%.6f %.6f %.6f' %(nodes[i][0], nodes[i][1], 
                                           nodes[i][2]))#, dists[i]))
        fOut.write('\n')
    print('%s file written' %outfile)
  return



def mainpath_color(geo, outfile, axon=None, things=None, interdist=1.):
  # This will color segments based on their distance from the mainpath
  print(axon)
  paths = np.array(axon_path(geo, axon, things))
  bpaths = []
  for path in paths:
    for p in path:
      bpaths.append(p)
  downpaths = bpaths[::20]
  segdists = []
  count = -1
  for s in geo.segments:
    count = count + 1
    if s in paths:
      segdists.append(0)
    else:
      pDF = PathDistanceFinder(geo, s)
      segdists.append(min([pDF.distanceTo(p) for p in downpaths]))
    if count % 10 == 0:
      print('%i (of %i) segments processed' %(count, len(geo.segments)))
  if len(segdists) != len(geo.segments):
    print('Num segdists (%i) not equal to num segments (%i)'
          %(len(segdists), len(geo.segments)))
    return
  pDF = PathDistanceFinder(geo, geo.soma)
  maxp = max(segdists)
  nodes, dists = [], []
  for s in geo.segments:
    if s.length > interdist:
      x0, y0, z0 = s.coordAt(0)[0],s.coordAt(0)[1], s.coordAt(0)[2]
      x1, y1, z1 = s.coordAt(1)[0],s.coordAt(1)[1], s.coordAt(1)[2]
      temp_nodes = [np.linspace(x0, x1, int(s.length/interdist)),
                    np.linspace(y0, y1, int(s.length/interdist)),
                    np.linspace(z0, z1, int(s.length/interdist))]
      #print(np.shape(temp_nodes))
      for t in range(np.shape(temp_nodes)[1]):
        nodes.append([temp_nodes[0][t],temp_nodes[1][t],temp_nodes[2][t]])
        dists.append(segdists[geo.segments.index(s)])
    else: # Add at least one node per segment
      nodes.append(s.coordAt(0.5))
      dists.append(segdists[geo.segments.index(s)])
  with open(outfile, 'w') as fOut:
    for n in range(len(nodes)):
      fOut.write('%.6f %.6f %.6f %.6f\n' %(nodes[n][0], nodes[n][1], 
                                         nodes[n][2], dists[n]))
  print('File %s written with %i nodes' %(outfile, len(nodes)))
  return




# Helper function for get_subtrees
def add_unidirect(geo, seg, prevseg, path):
  # Add all neighbors in a unidirectional manner.
  subtree = [] # This is a collection of segments in the subtree
               # There is no inherent structure to the subtree
  for n in seg.neighbors:
    if n != seg and n != prevseg and n not in path:
      subtree.append(n)
  same = 0
  while same < 5:
    prevlength = len(subtree)
    for s in subtree:
      for n in s.neighbors:
        if n != seg and n != prevseg and n not in subtree and n not in path:
          subtree.append(n)
    if len(subtree) == prevlength:
      same = same + 1
    else:
      same = 0
      print('subtree is length: %i' %len(subtree))
  return subtree



def get_subtrees(geo, axons, things=None):
  # This returns the subtrees coming off the 'main' path (soma->axon)
  # Should work with multiple axons.
  def is_subtree_root(geo, seg, path):
    subtree = []
    for n in seg.neighbors:
      if n not in np.array(path).flatten():
        subtree.append(add_unidirect(geo, n, seg, path)) ## 
    return subtree
  # Condition axon input
  paths = axon_path(geo, axons, things)
  flatpath = []
  for path in paths:
    for p in path:
      if p not in flatpath:
        flatpath.append(p)
  # Assume path[0] == geo.soma
  subtrees = []
  for p in flatpath:
    # Loop over the segments in the path to create the subtrees
    if len(p.neighbors) > 2:
      subs = is_subtree_root(geo, p, flatpath)
      for sub in subs:
        if sub not in subtrees:
          if len(sub) not in [len(i) for i in subtrees]:
            subtrees.append(sub)
  # Should have all unique subtrees now
  return subtrees



def show_subtrees(geo, outfile, axon=None, things=None, subtrees=None, interdist=1.):
  # Make a txt file to plot individual subtrees in matlab
  if subtrees is None:
    subtrees = get_subtrees(geo, axon, things)
    path = axon_path(geo, axon, things)
    subtrees.append(path)
  treenum = np.linspace(0,1,len(subtrees))
  # Interpolate for plotting
  nodes, dists = [], []
  for tree in range(len(subtrees)):
    for seg in subtrees[tree]:
      if type(seg) is list:
        for s in seg:
          if s.length > interdist:
            x0, y0, z0 = s.coordAt(0)[0],s.coordAt(0)[1], s.coordAt(0)[2]
            x1, y1, z1 = s.coordAt(1)[0],s.coordAt(1)[1], s.coordAt(1)[2]
            temp_nodes = [np.linspace(x0, x1, int(s.length/interdist)),
                          np.linspace(y0, y1, int(s.length/interdist)),
                          np.linspace(z0, z1, int(s.length/interdist))]
            #print(np.shape(temp_nodes))
            for t in range(np.shape(temp_nodes)[1]):
              nodes.append([temp_nodes[0][t],temp_nodes[1][t],temp_nodes[2][t]])
              dists.append(treenum[tree])
          else: # Add at least one node per segment
            nodes.append(s.coordAt(0.5))
            dists.append(treenum[tree])
      else:
        s = seg
        if s.length > interdist:
          x0, y0, z0 = s.coordAt(0)[0],s.coordAt(0)[1], s.coordAt(0)[2]
          x1, y1, z1 = s.coordAt(1)[0],s.coordAt(1)[1], s.coordAt(1)[2]
          temp_nodes = [np.linspace(x0, x1, int(s.length/interdist)),
                        np.linspace(y0, y1, int(s.length/interdist)),
                        np.linspace(z0, z1, int(s.length/interdist))]
          #print(np.shape(temp_nodes))
          for t in range(np.shape(temp_nodes)[1]):
            nodes.append([temp_nodes[0][t],temp_nodes[1][t],temp_nodes[2][t]])
            dists.append(treenum[tree])
        else: # Add at least one node per segment
          nodes.append(s.coordAt(0.5))
          dists.append(treenum[tree])
  with open(outfile, 'w') as fOut:
    for n in range(len(nodes)):
      fOut.write('%.6f %.6f %.6f %.6f\n' %(nodes[n][0], nodes[n][1], 
                                         nodes[n][2], dists[n]))
  print('File %s written with %i nodes' %(outfile, len(nodes)))
  return



### Wiring
def subtree_wiring(geo, subtrees=None, path=None, axons=None, things=None):
  # Returns the percent of total wiring of each subtree (and the path)
  if subtrees is None:
    ssubtrees = get_subtrees(geo, axons, things)
  else:
    ssubtrees = subtrees
  # Get rid of zero subtree
  pico = [len(t) for t in ssubtrees]
  if min(pico) == 0:
    ssubtrees.pop(pico.index(0))
  if path is None:
    paths = axon_path(geo, axons, things)
  else:
    paths = path
  pathwire = sum([sum([p.length for p in pa]) for pa in paths])
  allsegs = []
  for t in ssubtrees:
    for s in t: # Add each segment
      allsegs.append(s)
  total = sum([s.length for s in np.array(allsegs)])
  total = total + pathwire
  for p in paths:
    ssubtrees.append(p)
  subwiring = []
  for t in ssubtrees:
    subwiring.append(sum([s.length for s in t])/total)
  print('Subtree length: %i' %len(ssubtrees))
  print('Max: %.5f, Min: %.5f' %(max(subwiring), min(subwiring)))
  print('Mean: %.5f, Med: %.5f' %(np.mean(subwiring), np.median(subwiring)))
  print('Path wiring: %.5f, Path wiring percent: %.5f' %(pathwire, 
                                                         pathwire/total))
  print('Total tree wiring: %.5f' %total)
  print('Axons: '); print(axons)

  





### Subtree angles
def subtree_angles(subtrees, path, keep=10):
  # Returns the first N('keep') successive branch angles for each subtree.
  angles = []
  for t in subtrees:
    root = union(t, path) # Find the first non-mainpath segment
    count = 0
    prev_segs = [p for p in path]
    prev_segs.append(root)
    currsegs = [root]
    tree_angles = []
    while count < keep: # Keep going until 'keep' or run out of segs
      curr_angles = [] # Keep a list of angles for the current round
      next_segs = []
      for curr in currsegs:
        for n in curr.neighbors:
          if n not in prev_segs:
            pts = find_points(n, curr)
            curr_angles.append(get_angle(pts[0], pts[1], pts[2]))
            next_segs.append(n)
        prev_segs.append(curr)
      curr_segs = next_segs
      try:
        tree_angles.append(np.mean(curr_angles))
      except:
        count = 10
      count = count + 1
    # curr_angles should have <= 10 lists of angles, now these are averaged
    # angles.append([np.mean(o) for o in 
  return




## Still work in progress
def dendrogram_subtrees(geo, paths, subtrees):
  def plot_next(segs, prev_segs, x_range, prev_pts):
    if len(segs) != len(prev_pts):
      print('segs (%i) and prev_pts (%i) should be same length'
            %(len(segs), len(prev_pts)))
      return
    
  def subtree_pts(sub, x_range, paths):
    for s in sub:
      if s in np.array(paths).flatten():
        root = s
    
    return
    
  if len(paths) > 10:
    # There is only one path (one axon)
    m = len(subtrees)
  
  return
    
  



######################################################################
# partition asymmetry


def path_asymmetry(geo ):
  """
  This version uses paths and sets to calculate the length asymmetry.
  """
  pDF = PathDistanceFinder(geo, geo.soma, 0) # initialize pDF
  tipSegs, tipLocs = geo.getTipIndices()
  paths = [] # get all the paths
  for s in range(len(tipSegs)):
    try:
      paths.append(pDF.pathTo(geo.segments[tipSegs[s]],tipLocs[s]))
    except:
      continue
  length_asymmetry = []
  for p1 in paths:
    if paths.index(p1) % 100 == 0:
      print('completed %i (of %i) paths' %(paths.index(p1),len(paths)))
    for p2 in paths:
      if p1 != p2: # make sure not the same
        p_root = [i for i in p1 if i in p2]
        # make sure they share common segments, then do analysis
        if len(p_root) > 0:
          l_p1 = sum([s.length for s in p1])
          l_p2 = sum([s.length for s in p2])
          l_root = sum([s.length for s in p_root])
          if l_p1 > l_p2: # always put greater in denominator (customary)
            if l_p2 == 0:
              print('missed one')
              pass
            else:
              length_asymmetry.append((l_p2-l_root)/(l_p1-l_root))
          else:
            if l_p1 == 0:
              print('missed one')
            else:
              length_asymmetry.append((l_p1-l_root)/(l_p2-l_root))
  if len(length_asymmetry) > 1000:
    return length_asymmetry[::int(len(length_asymmetry)/1000)]
  else:
    return length_asymmetry



def get_segment(geo, segname):
  for s in geo.segments:
    if s.name == segname:
      return s


def add_all_downstream(geo, seg, prev_seg, tips):
  # print('called add_all_downstream')
  newsegs = [seg]
  prevsegs = [prev_seg]
  for n in prev_seg.neighbors:
    if n != seg:
      prevsegs.append(n)
  go = True
  same = 0
  while go:
    old_len = len(newsegs)
    for n in newsegs:
      for neb in n.neighbors:
        if neb not in prevsegs:
          newsegs.append(neb)
      prevsegs.append(n)
    if len(newsegs) == old_len: # if no change in newsegs, increment
      same = same + 1
    else:
      same = 0
    if same > 10: # after 10 no-changes, stop
      go = False
  # get tips and lengths
  pDF = PathDistanceFinder(geo, seg, 0)
  numtips, length = 0, []
  for t in tips:
    try:
      if geo.segments[t] in newsegs:
        numtips = numtips + 1
        add_segs = pDF.pathTo(t,1)
        for a in add_segs:
          if a not in length:
            length.append(a)
    except:
      continue
  length = sum([l.length for l in length])
  return length, numtips



def tips_asymmetry(geo):
  """
  Get the tip asymmetry of the neuron. Follow the soma's neighbors
  until there are more than 1, then start there.
  seg_lengths: dict with a section_name for keys and float as values
  seg_tips: dict with sec_name as key and list of segment objects as values
  """
  def get_bif_info(geo, seg, prev_seg, tips):
    # Given a branch and the previous branch (so we know the direction
    # of movement), make a dictionary with the non-previous neighbors as
    # keys and add all subsequent lengths and tips to the respective keys
    # print('called get_bif_info')
    forward_nebs = [n for n in seg.neighbors if n != prev_seg]
    neb_dict = {}
    lengths, tips = [], []
    for f in forward_nebs:
      length, tip = add_all_downstream(geo, f, seg, tips)
      # now it should be part, for future reference, too
      lengths.append(length)
      tips.append(tips)
      # neb_dict[f.name] = {'prevbranch': branch.name, 'length': 0, 'tips': 0}
    length_asym = []
    for l in lengths:
      try:
        length_asym.append(l/(sum(lengths)-l))
      except:
        continue
    #length_asym = [i/(sum(lengths)-i) for i in lengths]
    tip_asym = [] # [i/(sum(tips)-i) for i in tips]
    for t in tips:
      try:
        tip_asym.append(t/(sum(tips)-t))
      except:
        continue
    return length_asym, tip_asym
  
  # go through all branches in order
  tips, _ = geo.getTipIndices()
  #print('got tips')
  master_lengths, master_tips = [], []
  prevsegs = [geo.soma]
  newsegs = [i for i in geo.soma.neighbors if i not in prevsegs]
  print(newsegs)
  go, same = True, 0
  while go:
    old_len = len(newsegs)
    for n in newsegs:
      nebs = n.neighbors
      for k in nebs: # remove repeated segs
        if k in prevsegs:
          nebs.pop(nebs.index(k))
      if len(nebs) > 1: # now if there are > 1 neighbors (a branch)
        print('found a bifurcation!')
        for k in nebs:
          lengths, tips = get_bif_info(geo, n, k, tips)
          for i in lengths:
            master_lengths.append(i)
          for j in tips:
            master_tips.append(j)
          if k not in newsegs:
            newsegs.append(k) # done with k-th neighbor
      elif len(nebs) == 1:
        # if it's not a bifurcation, add it to prevsegs
        prevsegs.append(nebs[0])
      if n not in prevsegs:
        prevsegs.append(n) # done with n-th newseg
    if len(newsegs) == old_len:
      same = same + 1
      #print(same)
    else:
      same = 0
    if same > 1000:
      go = False
      return master_lengths, master_tips, prevsegs
    old_len = len(newsegs)
  # it should never get to this part, but in case it does
  return master_lengths, master_tips, prevsegs


def tip_coords(geo, seg_tips):
  # return x-y-z tuples for each tip; just use the (1) position of each tip seg
  tip_coords = {}
  for k in seg_tips.keys():
    tip_coords[k] = []
    for t in seg_tips[k]:
      tip_coords[k].append(t.coordAt(1))
  return tip_coords


def simplify_asymmetry(geo):
  # simplification of asymmetry data
  seg_lengths, seg_tips = tips_asymmetry(geo)
  sumlengths = sum([seg_lengths[k] for k in seg_lengths.keys()])
  sumtips = sum([len(seg_tips[k]) for k in seg_tips.keys()])
  lengths = [seg_lengths[k]/(sumlengths-seg_lengths[k]) for k in seg_lengths.keys()]
  tips = [float(len(seg_tips[k]))/float((sumtips-len(seg_tips[k]))) for k in seg_tips.keys()]
  return lengths, tips



def somatofugal_length(geo):
  print('Building lengths and tips...')
  prev_segs = [geo.soma]
  next_segs = [n for n in geo.soma.neighbors if n not in prev_segs]
  seg_lengths, seg_tips = {}, {}
  for n in next_segs:
    nlen, ntip = add_all_downstream(geo, n, geo.soma)
    seg_lengths[n.name] = nlen
    seg_tips[n.name] = ntip
  go, cnt = True, 0
  while go:
    old_len, new_next_segs = len(prev_segs), []
    for n in next_segs:
      for neb in n.neighbors:
        if neb not in prev_segs and neb not in next_segs:
          # now that have all the new segs, get their len & tips
          nlen, ntip = add_all_downstream(geo, neb, n)
          if neb.name in seg_lengths.keys() or neb.name in seg_tips.keys():
            print('Tried to add %s but is already there' %neb.name)
          else:
            seg_lengths[neb.name], seg_tips[neb.name] = nlen, ntip
          # add nebs to next_segs and n to prev_segs
          new_next_segs.append(neb)
      # put n into prev_segs
      prev_segs.append(next_segs.pop(next_segs.index(n)))
    # next_segs should be empty now
    next_segs = [i for i in new_next_segs]
    if len(prev_segs) == old_len:
      cnt = cnt + 1
    else:
      cnt = 0
    if cnt == 50 or len(prev_segs)==len(geo.segments):
      go = False
  print('Got %i (of %i) lengths and %i (of %i) tips)'
        %(len(seg_lengths), len(geo.segments),
          len(seg_tips), len(geo.segments)))
  return seg_lengths, seg_tips  



def tips_asymmetry_old(geo):
  """
  ############## OLD VERSION!!! #####################
  Get the tip asymmetry of the neuron. Follow the soma's neighbors
  until there are more than 1, then start there.
  seg_lengths: dict with a section_name for keys and float as values
  seg_tips: dict with sec_name as key and list of segment objects as values
  """
  seg_lengths, seg_tips = somatofugal_length(geo) # this will get a lot of them
  def get_bif_info(geo, seg, prev_seg, seg_lengths, seg_tips):
    # Given a branch and the previous branch (so we know the direction
    # of movement), make a dictionary with the non-previous neighbors as
    # keys and add all subsequent lengths and tips to the respective keys
    # print('called get_bif_info')
    forward_nebs = [n for n in seg.neighbors if n != prev_seg]
    neb_dict = {}
    lengths, tips = [], []
    for f in forward_nebs:
      # if this seg isn't already part of seg_lengths/tips, add it
      if f.name not in seg_lengths.keys():
        length, tip = add_all_downstream(geo, f, seg)
        seg_lengths[f.name] = length
      if f.name not in seg_tips.keys():
        seg_tips[f.name] = tip
      # now it should be part, for future reference, too
      lengths.append(seg_lengths[f.name])
      tips.append(seg_tips[f.name])
      # neb_dict[f.name] = {'prevbranch': branch.name, 'length': 0, 'tips': 0}
    length_asym = []
    for l in lengths:
      try:
        length_asym.append(l/(sum(lengths)-l))
      except:
        continue
    #length_asym = [i/(sum(lengths)-i) for i in lengths]
    tip_asym = [] # [i/(sum(tips)-i) for i in tips]
    for t in tips:
      try:
        tip_asym.append(t/(sum(tips)-t))
      except:
        continue
    return length_asym, tip_asym, seg_lengths, seg_tips
  
  # go through all branches in order
  master_lengths, master_tips = [], []
  prevsegs = [geo.soma]
  newsegs = [i for i in geo.soma.neighbors if i not in prevsegs]
  # print(newsegs)
  go, same = True, 0
  while go:
    old_len = len(newsegs)
    for n in newsegs:
      nebs = n.neighbors
      for k in nebs: # remove repeated segs
        if k in prevsegs:
          nebs.pop(nebs.index(k))
      if len(nebs) > 1: # now if there are > 1 neighbors (a branch)
        print('found a bifurcation!')
        for k in nebs:
          lengths, tips, seg_lengths, seg_tips = get_bif_info(geo, n, k, seg_lengths, seg_tips)
          for i in lengths:
            master_lengths.append(i)
          for j in tips:
            master_tips.append(j)
          if k not in newsegs:
            newsegs.append(k) # done with k-th neighbor
      if n not in prevsegs:
        prevsegs.append(n) # done with n-th newseg
    if len(newsegs) == old_len:
      same = same + 1
    else:
      same = 0
    if same > 10:
      go = False
      return master_lengths, master_tips, prevsegs
    old_len = len(newsegs)
  # it should never get to this part, but in case it does
  return master_lengths, master_tips, prevsegs
  


######################################################################
# torques

def getNormVector(points):
  #print(points, np.shape(points))
  v1 = [points[1][0][i] - points[0][0][i] for i in range(3)]
  v2 = [points[2][0][i] - points[0][0][i] for i in range(3)]
  normVec = np.cross(v1,v2)
  return normVec


def angleBetween(plane1,plane2,planCoords):
  # get normal vectors
  n1, n2 = getNormVector(planCoords[plane1]), \
           getNormVector(planCoords[plane2])
  angle = np.arccos( (abs(n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2])) /
                     ( np.sqrt(n1[0]**2+n1[1]**2+n1[2]**2) *
                       np.sqrt(n2[0]**2+n2[1]**2+n2[2]**2) ) )
  return angle*180/np.pi


def get_torques(geo):
  # return bifurcation torques
  Cons =  geo.connections
  Seg1s, Seg2s = [], []
  for c in Cons:
    Seg1s.append(c['filament1']) # here, location1 is always 0
    Seg2s.append(c['filament2']) # here, location2 is always 1
    #geometry.c['filament1'].coordAt(c['location1'])
  
  tsegs = np.array([Seg1s,Seg2s]).T
  tsegs = tsegs.reshape(len(tsegs)*2)
  segs = set(tsegs)
  planCoords = {}
  
  count = 0
  for seg in segs:
    friends, friendcoords = [], []
    for s in geo.segments:
      if s.name == seg:
        friends.append(s.name)
        if s.name in Seg1s:
          friends.append(Seg2s[Seg1s.index(s.name)])
        if s.name in Seg2s:
          friends.append(Seg1s[Seg2s.index(s.name)])
    #print('friends compiled')
      
    for s in geo.segments:
      if s.name in friends:
        friendcoords.append([s.coordAt(1)])
    count = count + 1
    #if count%100 == 0:
    #  print('%i of %i segments done' %(count, len(segs)))
    if len(friendcoords) > 2: # need 3 points to define plane
      planCoords[seg]=friendcoords
  
  planCoordskeys = []
  for s in geo.segments: # loop through segments to find plane-neighbors
    if s.name in planCoords.keys():
      for n in s.neighbors:
        if n.name in planCoords.keys(): # if the neighbor is also a bifurcation
          planCoordskeys.append([s.name, n.name]) # add it
        else: # otherwise, keep looking for a neighbor that is
          for nn in n.neighbors:
            if nn.name in planCoords.keys():
              planCoordskeys.append([s.name, nn.name])
  
  # get torques
  torques = []
  for P in planCoordskeys:
    torques.append(angleBetween(P[0],P[1],planCoords))
  return torques



###############################################################################
# Ellipse fitting, distance to nearest point stuff


def getNoSomaPoints(geo):
  # get the downsampled nodes, but not the soma
  somaPos = geo.soma.coordAt\
            (geo.soma.centroidPosition(mandateTag='Soma'))
  print('Soma position: %.5f, %.5f, %.5f' %(somaPos[0],somaPos[1],somaPos[2])) # works
  nodes = []
  for seg in geo.segments:
    if 'Soma' not in seg.tags:
      nodes.append(seg.coordAt(0))
      nodes.append(seg.coordAt(0.5))
      nodes.append(seg.coordAt(1))
  print('Sampled %i nodes' %len(nodes))
  
  return nodes



def findBounds(nodelist):
  # return the x,y,z bounds of the node list
  xs, ys, zs = [], [], []
  
  for n in range(len(nodelist)):
    xs.append(nodelist[n][0])
    ys.append(nodelist[n][1])
    zs.append(nodelist[n][2])

  bounds = {'xmin': min(xs), 'xmax': max(xs), 
            'ymin': min(ys), 'ymax': max(ys),
            'zmin': min(zs), 'zmax': max(zs)}
  
  return bounds



def getGridPoints(nodelist, pplot=False):
  # create a grid around the neuropil and use linspace to fill the volume
  bounds = findBounds(nodelist)
  gridpoints = []
  xs = np.linspace(bounds['xmin'], bounds['xmax'], 10)
  ys = np.linspace(bounds['ymin'], bounds['ymax'], 10)
  zs = np.linspace(bounds['zmin'], bounds['zmax'], 10)
  spacing = xs[1]-xs[0]
  
  # 1000 grid volume points
  for i in range(len(xs)-1):
    for j in range(len(ys)-1):
      for k in range(len(zs)-1):
        gridpoints.append([(xs[i]+xs[i+1])/2,
                           (ys[j]+ys[j+1])/2,
                           (zs[k]+zs[k+1])/2])
  print('gridpoints is length %i' %len(gridpoints))
  
  boxx, boxy, boxz = [], [], []
  for g in range(len(gridpoints)):
    boxx.append(gridpoints[g][0])
    boxy.append(gridpoints[g][1])
    boxz.append(gridpoints[g][2])
  
  nodex, nodey, nodez = [], [], []
  for n in range(len(nodelist)):
    nodex.append(nodelist[n][0])
    nodey.append(nodelist[n][1])
    nodez.append(nodelist[n][2])
  
  if pplot:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
  #ax.plot(boxx, boxy)
    ax1.scatter(boxx, boxy, boxz, color='r', marker='.', alpha=0.5)
    ax1.scatter(nodex, nodey, nodez, color='k', marker='.', alpha=1)
  # ax.set_xlabel('')
  # plt.show()
    
  return gridpoints, spacing
  


def closestPoint(rectpoint, nodes):
  # find the closest neuron node to a rectangle point
  ptmin = np.inf
  ptind, pt = None, None
  for n in range(len(nodes)):
    dist = dist3(rectpoint, nodes[n])
    if dist < ptmin:
      ptmin = dist
      ptind = n
      pt = nodes[n]
  return ptind, ptmin


def closestPointPool(things):
  # find the closest neuron node to a rectangle point
  # things[0] = rect point, things[1] = all nodes
  things[0] = rectpoint
  things
  ptmin = np.inf
  ptind, pt = None, None
  for n in range(len(nodes)):
    dist = dist3(rectpoint, nodes[n])
    if dist < ptmin:
      ptmin = dist
      ptind = n
      pt = nodes[n]
  return ptind, ptmin


def getSurfacePoints(gridpoints, nodes, spacing, pplot=False):
  # given volume points and neuropil nodes, create downsampled
  # volume of the neuropil (if a neuron point is in a given cube, 
  # the cube is a 1, else 0
  ellipsePoints = []
  if type(gridpoints) is not np.ndarray:
    gridpoints = np.array(gridpoints)
  if type(nodes) is not np.ndarray:
    nodes = np.array(nodes)
  
  for b in range(len(gridpoints)):
    _, dist = closestPoint(gridpoints[b], nodes)
    if dist <= spacing/8.:
      ellipsePoints.append(gridpoints[b])
    if b % 100 == 0 and b != 0:
      print('%i/%i points examined' %(b, len(gridpoints)))
      
  print('Now have %i neuropil points' %len(ellipsePoints))
  
  surfx, surfy, surfz = [], [], []
  for s in ellipsePoints:
    surfx.append(s[0])
    surfy.append(s[1])
    surfz.append(s[2])
  if pplot:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(surfx, surfy, surfz, color='g', marker='.')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    plt.show()
  
  return ellipsePoints


def writeFile(points, outfile):
  # write points to a ascii; this is generally not necessary
  if outfile is None:
    outfile = 'neuropil_surfpoints.txt'  
  with open(outfile, 'w') as fOut:
    for p in range(len(points)):
      # print(points[p])
      ptstring = [str(points[p][0]), str(points[p][1]), str(points[p][2])]
      ptstr = ' '.join(ptstring)
      fOut.write(ptstr)
      fOut.write('\n')
      #print
  fOut.close()
  print('%s file written.' %outfile)
  return


# Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1

def give_ellipse(axes, shrink, translate):
  """
  axes: [1x3], shrink: scalar (ratio), translate: [1x3]
  Returns a 2-D ellipse of points when given the 3 axes ([maj, min, 3rd])
  and where on the 3rd axis the current slice is
  --> axes = original evals ### scale omitted here
  --> 'shrink' is the ratio that determines 
      how large how the ellipse should be stretched in 2-D
  --> axes[2] not used in this version
  """
  norm_ax = [i/max(axes) for i in axes]
  xs = np.linspace(-norm_ax[0],norm_ax[0],1000)
  ys = [np.sqrt( (1 - (i**2/norm_ax[0])) * norm_ax[1] ) for i in xs]
  # get rid of the nans
  opts = [[x,y] for x,y in zip(xs,ys) if np.isfinite(y)]
  # need to get the negative part of the y half of the graph
  pts = []
  for p in opts:
    pts.append([p[0],-p[1]])
    pts.append(p)
  # pts are currently the 'largest' possible, need to shrink by 'where'
  pts = np.array(pts)
  pts = pts * shrink
  newpts = []
  for p in pts:
    _pt = [axes[0] * p[0] + translate[0],  \
           axes[1] * p[1] + translate[1],  \
           translate[2]]
    if _pt not in newpts:
      newpts.append(_pt)
  
  return newpts


def get_reduced_points(geo, outfile=None):
  # only pre-req is to run getNoSomaPoints first
  nodes = getNoSomaPoints(geo)
  gridpoints, spacing = getGridPoints(nodes)
  ellipsePoints = getSurfacePoints(gridpoints, nodes, spacing)
  #writeFile(ellipsePoints, outfile)
  
  return ellipsePoints


def check_eigen(s_vals, s_vecs, pts):
  """
  For singular value decomposition, check the orientations of vectors
  vs. the points they're supposed to represent
  """
  # Get zero-centered points first
  #means = [pts[i] for i in range(len(pts)) if i%100==0] # downsample
  means = pts
  _m = [np.mean([j[0] for j in means]), np.mean([j[1] for j in means]),
        np.mean([j[2] for j in means])]
  # subtract the mean but keep the shape
  newmeans = []
  for m in means:
    newmeans.append([m[0]-_m[0],m[1]-_m[1],m[2]-_m[2]])
  dmax = farthest_pt(pts)
  # get eigenvectors normalized by distance from farthest pts
  scales = [i/max(s_vals)*dmax for i in s_vals]
  print(scales)
  v1 = [[0,0,0],[scales[0]*s_vecs[0][0], scales[0]*s_vecs[1][0], 
                 scales[0]*s_vecs[2][0]]]
  v2 = [[0,0,0],[scales[1]*s_vecs[0][1], scales[1]*s_vecs[1][1], 
                 scales[1]*s_vecs[2][1]]]
  v3 = [[0,0,0],[scales[2]*s_vecs[0][2], scales[2]*s_vecs[1][2], 
                 scales[2]*s_vecs[2][2]]]
  print(v1,v2,v3)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  for m in newmeans:
    ax.scatter(m[0],m[1],m[2], c='b', edgecolor='b', alpha=0.2)
  ax.plot([0,v1[1][0]], [0,v1[1][1]], [0,v1[1][2]], c='r')
  ax.plot([0,v2[1][0]], [0,v2[1][1]], [0,v2[1][2]], c='g')
  ax.plot([0,v3[1][0]], [0,v3[1][1]], [0,v3[1][2]], c='k')
  plt.show()
  return newmeans


def build_ellipse(geo):
  """
  Uses singular values from a uniformly resampled neuron grid to get
  major/minor axes to create an ellipsoid; scales and translates the
  ellipsoid back to neuron space.
  """
  gpts = get_reduced_points(geo)
  gmean = [np.mean([i[0] for i in gpts]),
           np.mean([i[1] for i in gpts]),
           np.mean([i[2] for i in gpts])]
  # get singular values
  _, s_vals, s_vecs = np.linalg.svd(gpts)
  s = np.array([i/max(s_vals) for i in s_vals])
  # scale singular values by longest distance
  dmax = farthest_pt(gpts)
  s = s * dmax
  # hyperbolic scaling reference for taper of top/bottom
  _x = np.linspace(0,10,50)
  _y = -_x**2 + 100
  y = [i/max(_y) for i in _y]
  y.reverse()
  zscale = [i for i in y]
  y.reverse()
  for i in y:
    zscale.append(i)
  eig_pts = []
  # make 100 layers of v3
  zlayers = np.linspace(-s[2],s[2],100)
  for v in zlayers:
    newpts = give_ellipse(s, zscale[list(zlayers).index(v)], 
                          [0,0,0])
    for p in newpts:
      eig_pts.append(p)
  eig_pts = np.array(eig_pts)
  # now have all eigen points, need to re-orient axes
  pts = eig_pts.dot(np.linalg.inv(s_vecs))
  # now translate:
  pts = [[p[0]+gmean[0], p[1]+gmean[1], p[2]+gmean[2]] for p in pts]
  return pts, gpts, eig_pts
  
    
def get_distances(geo, multi=None):
  """
  Return the "distances", the distance from each ellipse point to the
  closest point of the neuron's skeleton. Can be parallelized by multi=int.
  """
  if multi is None:
    ellipse_pts, _, _ = build_ellipse(geo)
    nodes = getNoSomaPoints(geo)
    distances = []
    # make sure this isn't ridiculously long
    if len(ellipse_pts) > 100000:
      ellipse_pts = ellipse_pts[::10]
    for e in ellipse_pts:
      _, d = closestPoint(e, nodes)
      distances.append(d)
      if ellipse_pts.index(e)%1000==0:
        print('%i (of %i) sampled so far' %(ellipse_pts.index(e), len(ellipse_pts)))
    return distances
  elif type(multi) is int:
    from multiprocessing import Pool
    p = Pool(multi)
    #distances = pool.map(closestPointPool, 
  return distances
  


#######################################################################
# simple branch stuff

def branch_lengths(geo, locations=False):
  lengths = [b.length for b in geo.branches]
  locations = [b.coordAt(0.5) for b in geo.branches]
  if locations:
    return lengths, locations
  else:
    return lengths


def branch_order(geo):
  geo.calcForewardBranchOrder()
  return [b.branchOrder for b in geo.branches]
    

def length_vs_dist(geo=None, lengths=None, locations=None):
  # Simple calculation of branch lengths
  if lengths is None and locations is None and geo is not None:
    lengths, locations = branch_lengths(geo, True)
  midpt = [np.mean([i[0] for i in locations]),
           np.mean([i[1] for i in locations]),
           np.mean([i[2] for i in locations])]
  print('Calculating distances....')
  distances = [dist3(pt, midpt) for pt in locations]
  return lengths, distances
    
  
# Multiple-regress plot
def multi_regress(lol):
  # lol must be a list of lists (each item is a list of 2 lists)
  return




#######################################################################
# tip-to-tip distances

def tip_to_tip(geo):
  """
  Who knows -- this might be important some day.
  """  
  tips, tipInds = geo.getTipIndices()
  tip_dists = []
  Tips = list(zip(tips, tipInds))
  pDF = PathDistanceFinder(geo, geo.segments[tips[0]], tipInds[0])
  for t in range(len(Tips)):
    try:
      tip_dists.append(pDF.distanceTo(geo.segments[Tips[t][0]], Tips[t][1]))
    except:
      print('missed one')
  return tip_dists
  




#######################################################################
# fractal dimension (as per Caserta et al., 1995)
# For every point in the (resampled/interpolated) neuron, basically do
# a center of mass calculation for 'radius of gyration'
#  (doesn't need to be interpolated -- sampled from every neuron point)

# helper functions
def element_histogram(data, bins):
  # Also returns the actual data, list (of len(bins)-1) of lists
  # This might be bottle neck -- omitted from function for now
  thing = [[] for i in bins]
  noplace = 0
  for d in data:
    for b in range(len(bins)):
      if d >= bins[-1]: # this should never really happen except maybe once
        thing[-1].append(d)
      elif d <= bins[0]:
        thing[0].append(d)
      elif d >= bins[b] and d < bins[b+1]:
        thing[b].append(d)
      else:
        noplace = noplace + 1
  if len(thing[-1]) > 0:
    print('lost %i points from last bin' %len(thing[-1]))
  print('Could not place %i (of %i) points' %(noplace, sum([len(i) for 
                                              i in thing])))
  return thing[:-1]


def pt8_slope(bins, masses, where=False):
  # Get the average constant 8-pt slope
  if len(bins) > len(masses):
    bins = bins[:len(masses)]
  if len(masses) > len(bins):
    masses = masses[:len(bins)]
  bins, masses = [np.log(i) for i in bins], [np.log(i) for i in masses]
  slopes = [(masses[i+1]-masses[i])/(bins[i+1]-bins[i]) for i in
                                                        range(len(bins)-1)]
  delta = [abs(slopes[i+1]-slopes[i]) for i in range(len(slopes)-1)]
  # find 8-pt min for delta in slope
  mins = [sum(delta[i:int(i)+7]) for i in range(len(delta)-8)]
  possible_pts = [i for i in mins]
  possible_pts.sort()
  for p in possible_pts:
    check = np.mean(slopes[mins.index(p):mins.index(p)+7])
    if check > 0:
      start_pt = mins.index(p)
      break
  frac_dim = np.mean(slopes[start_pt:start_pt+7])
  if where == True:
    return frac_dim, start_pt+3
  return frac_dim


# Fractal dimension
def fractal_dimension(geo, where=False):
  """
  Calculate fractal dimension.
  """
  pts = [nodex(n) for n in geo.nodes]
  if len(pts) > 10000:
    div = int(len(pts)/10000)
    pts = pts[::div] # downsample to ~ 10,000 pts for time
  maxm = farthest_pt(pts)
  bin_e = np.linspace(0., maxm, 1001)
  dists = []
  for i in pts:
    if pts.index(i)%1000==0:
      print('%i (of %i) centers analyzed' %(pts.index(i), len(pts)))
    for j in pts:
      dists.append(dist3(i,j))
  # now all dists are calculated
  hist, bin_e = np.histogram(dists, bin_e)
  #spheres = element_histogram(dists, bin_e)
  bins = [(bin_e[i]+bin_e[i+1])/2 for i in range(len(bin_e)-1)]
  masses = [np.sqrt((bins[i]**2 * hist[i]) / np.sqrt(hist[i])) 
            for i in range(len(bins))] # radius of gyration
  # if hist[i] == 0 this gives nan, clean these date
  bad_inds = [i for i in range(len(masses)) if str(masses[i])=='nan']
  masses = [masses[i] for i in range(len(masses)) if i not in bad_inds]
  bins = [bins[i] for i in range(len(bins)) if i not in bad_inds]
  # should now be a mass for every bin radius
  frac_dim = pt8_slope(bins, masses, where)
  if type(frac_dim) is list:
    return frac_dim[0], bins, masses, frac_dim[1]
  return [frac_dim, bins, masses]
  

# plot fit
def showFractalDimension(frac_d, bins, masses, fname=None):
  # Show a plot of the fit to the constant slope
  frac_dim, loc = pt8_slope(bins, masses, where=True)
  #if bins2 != bins or masses2 != masses:
  #  print('Discrepency in new elements of bins or masses; using old values')
  ptx, pty = [np.log(bins[0]), np.log(bins[loc]), np.log(bins[-1])], []
  pty.append(np.log(masses[loc])-(np.log(bins[loc])-np.log(bins[0]))*frac_dim)
  pty.append(np.log(masses[loc]))
  pty.append(np.log(masses[loc])+(np.log(bins[-1])-np.log(bins[loc]))*frac_dim)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(np.log(bins), np.log(masses), color='b', edgecolor='b',
             alpha=0.2)
  ax.plot(ptx, pty, color='r', linewidth=3, alpha=0.5)
  ax.set_xlabel('Log radius um^x')
  ax.set_ylabel('Log mass')
  if fname is not None:
    ax.set_title('Fractal dimension for %s (%.2f)' %(fname, frac_dim))
  else:
    ax.set_title('Fractal dimension (%.2f)' %frac_dim)
  plt.show()
  return 
  


#######################################################################
# soma position
# this is kind of a bitch, only way I can think is the show the user
# and let them decide; could automate and just take opposite of axon ,
# but the program may find the axon wrong and then don't want to compound
# mistakes; also, GM projects to aln and this won't work for that

def display_simple_neuron(geo):
  # default is to show lots of soma and axon but little of everything else
  pts = [] # populate segments
  for s in geo.segments:
    pts.append(s.coordAt(0.5))
  axons = geo.findAxons() # populate axon
  axs = []
  for a in axons:
    for n in a.nodes:
      axs.append([n.x,n.y,n.z])
  sms = [] # populate soma
  for n in geo.soma.nodes:
    sms.append([n.x,n.y,n.z])
  # plotting shit
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for p in pts:
    ax.scatter(p[0],p[1],p[2], c='b',edgecolor='b', alpha=0.2)
  for p in axs:
    ax.scatter(p[0],p[1],p[2], c='r',edgecolor='r', alpha=0.5)
  for p in sms:
    ax.scatter(p[0],p[1],p[2], c='k', edgecolor='k', alpha=0.5)
  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')
  ax.set_zlabel('z axis')
  ax.set_aspect('equal')
  plt.show()
  return



def place_soma(geo, stn_val):
  """
  Places the stn in 3-D to find the soma's position. stn_val should be
  a 3-tuple where any None value becomes the mean for that coordinate;
  i.e.: for 836_047 the y and z are none but the x is -100 to place the 
  stn at the end of the neuropil, so stn_val = [-100, None, None]
  This function also assumes an X-Y- major projection! Important for phi.
  """
  # These copied from branch_angles
  def dist3(pt0, pt1):
    if len(pt0) == len(pt1) and len(pt0) == 3:
      return math.sqrt(sum([(pt0[i]-pt1[i])**2 for i in range(3)]))
    else:
      print('dimension mismatch')
      print(pt0, pt1)
  #
  def get_angle(pt0, midpt, pt1):
    if pt0 in [midpt, pt1] or pt1 in [midpt, pt0] or midpt in [pt0,pt1]:
      print('Some points are the same!')
      print(pt0, midpt, pt1)
    PT0 = dist3(pt1, midpt)
    PT1 = dist3(pt0, midpt)
    MIDPT = dist3(pt0, pt1)
    try:
      ang = math.acos( (MIDPT**2 - PT1**2 - PT0**2) / (2*PT1*PT0) )
      ang = ang*180/math.pi
    except:
      ang = 'nan'
    return ang
  #
  pts = [s.coordAt(0) for s in geo.segments]
  means = [np.mean([m[0] for m in pts]),
           np.mean([m[1] for m in pts]),
           np.mean([m[2] for m in pts])]
  for s in range(len(stn_val)): # make sure stn_val is kosher
    if stn_val[s] == None or stn_val[s] == 0:
      stn_val[s] = means[s]
  theta_val = stn_val
  phi_val = [means[0],means[1],100] # doesn't really matter for z, just needs be positive
  soma_val = geo.soma.coordAt(0)
  theta = get_angle(theta_val, means, soma_val)
  phi = abs(get_angle(phi_val, means, soma_val))-90. # from X-Y plane
  r = dist3(means, soma_val)
  return 180-theta, phi, r



def plot_soma_positions(arr, types=None):
  """
  Input is a list of 3-tuples [theta (from stn in X-Y), phi (elevation
  from X-Y plane at stn and center of neuropil), r (distance from center 
  of neuropil].
  """
  def polar_to_rect(p):
    x = p[2]*np.cos(p[0]/180*np.pi)
    y = p[2]*np.sin(p[0]/180*np.pi)
    z = p[2]*np.sin(p[1]/180*np.pi)
    return [x,y,z]
  pts = [polar_to_rect(p) for p in arr]
  stn_length = max([p[2] for p in arr])
  # get colors
  colors = ['darkkhaki','royalblue','forestgreen','tomato']
  if types:
    if type(types[0]) is not int:
      names = list(set(types))
      types = [names.index(i) for i in types]
    cols = [colors[i] for i in types]
    patches = []
    for n in range(len(names)): # should not exceed 3 (4 cell types)
      patches.append( mpatches.Patch(color=colors[n], label=names[n]) )
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # set up the axes
  ax.plot([0,stn_length], [0,0],[0,0], linewidth=5, c='k')
  ax.plot([0,-stn_length*.5],[0,0], [0,0], linewidth=1, c='k')
  ax.plot([0,0],[stn_length*.5,-(stn_length*.5)],[0,0], linewidth=1,c='k')
  ax.text(1.1*stn_length, 0, -20, r'stn', style='italic', fontsize=20)
  stnlen = '%.0f um' %(stn_length)
  ax.text(1.1*stn_length, 0, -50, stnlen, fontsize=15)
  # plot the soma
  for p in pts:
    if types:
      ax.scatter(p[0],p[1],p[2],s=100,c=cols[pts.index(p)], 
                 edgecolor=cols[pts.index(p)])
    else:
      ax.scatter(p[0],p[1],p[2],s=100)
  if types:
    plt.legend(handles=patches, loc='best')
  ax.set_zlim([-stn_length*.5, stn_length*0.5])
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  plt.show()
  return
  


































