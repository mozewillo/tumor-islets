import math
import shapely.geometry as geometry
from descartes import PolygonPatch
import pylab as plt
import numpy as np
from ctypes import byref, c_void_p, c_double

from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString, asMultiLineString, Point
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
import itertools

def plot_polygon(polygon):
    fig = plt.figure(figsize=[16, 9])
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#ffdede', ec='#ffdede', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig, patch

def shapeup(ob):
    if isinstance(ob, BaseGeometry):
        return ob
    else:
        try:
            return asShape(ob)
        except ValueError:
            return asLineString(ob)

def my_polygonize(lines):
    """Creates polygons from a source of lines
    The source may be a MultiLineString, a sequence of LineString objects,
    or a sequence of objects than can be adapted to LineStrings.
    """
    #source = getattr(lines, 'geoms', None) or lines
    #try:
    #     source = iter(source)
    #except TypeError:
    #     source = [source]
    #finally:
    obs = [shapeup(l) for l in lines.geoms] #source]
    geom_array_type = c_void_p * len(obs)
    geom_array = geom_array_type()
    for i, line in enumerate(obs):
        geom_array[i] = line._geom
    product = lgeos.GEOSPolygonize(byref(geom_array), len(obs))
    collection = geom_factory(product)
    for g in collection.geoms:
        clone = lgeos.GEOSGeom_clone(g._geom)
        g = geom_factory(clone)
        g._other_owned = False
        yield g



def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    @param only_outer:
    """
    
 
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    coords = np.array([point for point in points])
    tri = Delaunay(coords)
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(math.fabs(s * (s - a) * (s - b) * (s - c)))
        circum_r = a * b * c / (4.0 * area) if area > 0 else 999
        # Here's the radius filter.
        if circum_r < alpha:
            edge_points.extend([coords[[ia, ib]], coords[[ib, ic]], coords[[ic, ia]]])
    m = geometry.MultiLineString(edge_points)
    ### Here we put my_polygonize instead of shapely.polygonize due to Shapely 2.0 warning
    triangles = list(my_polygonize(m))
    return unary_union(triangles), edge_points
