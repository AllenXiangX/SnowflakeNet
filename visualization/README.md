### Visualization code for point splittings

----

The file *vis_splitting.py* contains code for visualizing point splitting paths. There are three import functions to use:

#### 1. splitting_paths

this function is for visualizing splitting paths between two point clouds, pcd2 should split from pcd1, and 'inds' is a list of point indices to visualize. The function will return an open3d mesh object of pcd1 and the specific splitting paths.

#### 2. splitting_paths_triple

this function is for visualizing two steps splitting for points specified in 'inds'.

#### 3. splittings_by_range

this function is for visualizing two steps splitting for points specified by ranges. If you want to visualize points 
of a small area in 3D space and don't know the exact point indices, this function will help you. Ranges are determined
by the visualization boundary on each axis (i.e. range_x, range_y, range_z), and ranges should be within [0, 1]. 

#### visualizing and saving mesh object.
All the above three function return open3d mesh object, use
```angular2html
open3d.visualization.draw_geometries([mesh])
```
to save mesh object, use
```angular2html
o3d.io.write_triangle_mesh(filename, mesh)
```
