#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: very_very_orienteering.py
#  Purpose: Very-Very-Orienteering Map Generator
#   Author: Tom Richter
#    Email: lorchel@gmx.de
#  License: GPLv3
#
# Copyright (C) 2012 Tom Richter
#---------------------------------------------------------------------
"""
Very-Very-Orienteering Map Generator
====================================

This is a Very-Very-Orienteering Map Generator. If you don't know these kind of
orienteering training: Try it out! It's fun!

It works like an window or compass orienteering with rotated windows. That means
after finding the control by running in its direction, the runner has to
observe the surroundings and find out which direction is North or rather in
which direction is the next control.
 
To generate your map just export your orienteering map to an image file and use
the filename as argument. Then create some controls in the map by simply
clicking. After that rotate the controls in the other window by dragging the
connecting lines.
The maps can be saved in a tar file with the same name as the image hitting
Shift-S. Inside the tar you'll find:
   - the original map
   - a full training map with your route
   - a training map with windows around the controls
   - the very-very-orienteering map
   - and some pickled data (for loading the session later)

The session can be loaded again by giving the tar filename as argument.
Be careful, because it will be overridden when changes are saved via Shift-S.
"""

#TODO Bug with indexing in plot_map2 -> Sometimes the indexes returned by
#     ind_boxes don't have the same size.
#TODO Line width now changes with figure size -> not practical at all when
#     exporting maps
#TODO Possibility to automatically put last control on start with button or
#     checkbox
#TODO Possibility to create shortcuts  

from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon, Patch
from matplotlib.widgets import Slider
from numpy import arctan2, pi
from scipy.ndimage.interpolation import rotate
from time import time
import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import tempfile
try:
    import cPickle as pickle
except ImportError:
    import pickle

_TIME_CLICK = 0.2

class Point(collections.namedtuple('Point', 'x y')):
    """Class representing a point"""
    __slots__ = ()
    def __add__(self, p2):
        try:
            p = Point(x=self.x + p2.x, y=self.y + p2.y)
        except AttributeError:
            p = Point(x=self.x + p2, y=self.y + p2)
        return p
    def __sub__(self, p2):
        try:
            p = Point(x=self.x - p2.x, y=self.y - p2.y)
        except AttributeError:
            p = Point(x=self.x - p2, y=self.y - p2)
        return p
    def __mul__(self, q):
        return Point(x=q * self.x, y=q * self.y)
    def __div__(self, q):
        return Point(x=self.x / q, y=self.y / q)
    def int_(self):
        return Point(x=int(round(self.x)), y=int(round(self.y)))

def slope(p1, p2):
    """Slope between two points"""
    m = arctan2(-(p2.y - p1.y), p2.x - p1.x)
    return (-m - pi / 2) % (2 * pi)
def distance(p1, p2):
    """Distance between two points"""
    return ((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2) ** 0.5
def on_way(p1, p2, q):
    """
    Point on line between p1 and p2 with relative distance q from p1
    
    q=0 -> Point is on p1
    q=1 -> Point is on p2
    Arguments:
    p1, p2: Points
    q: float 
    """
    return (p2 - p1) * q + p1
def rad(r, phi):
    """Point from polar coordinates"""
    return Point(-r * np.sin(phi), r * np.cos(phi))
def confine(p1, p2, size):
    """
    Returns corrected points p1, p2 that none is laying outside the window.
    """
    miny = min(p1.y, p2.y)
    maxy = max(p1.y, p2.y) - size.y
    minx = min(p1.x, p2.x)
    maxx = max(p1.x, p2.x) - size.x
    if miny < 0:
        p1 = Point(x=p1.x, y=p1.y - miny)
        p2 = Point(x=p2.x, y=p2.y - miny)
    if maxy > 0:
        p1 = Point(x=p1.x, y=p1.y - maxy)
        p2 = Point(x=p2.x, y=p2.y - maxy)
    if minx < 0:
        p1 = Point(x=p1.x - minx, y=p1.y)
        p2 = Point(x=p2.x - minx, y=p2.y)
    if maxx > 0:
        p1 = Point(x=p1.x - maxx, y=p1.y)
        p2 = Point(x=p2.x - maxx, y=p2.y)
    return p1, p2

def get_line_data(p1, p2, radius, radius2=None):
    """
    Returns plotable line data between Points p1 and p2
    
    The line is shortened by radius and radius2 (default: same as radius)
    """
    dist = distance(p1, p2)
    if radius2 is None:
        radius2 = radius
    return zip(on_way(p1, p2, radius / dist), on_way(p1, p2, 1 - radius2 / dist))

def ind_boxes(size, center1, center2, radius):
    """Indices of two boxes of same size fitting in the whole window"""
    ll1 = (center1 - radius).int_()
    ur1 = (center1 + radius).int_()
    ll2 = (center2 - radius).int_()
    ur2 = (center2 + radius).int_()
    ll1, ll2 = confine(ll1, ll2, size)
    ur1, ur2 = confine(ur1, ur2, size)
    return (slice(ll1.y, ur1.y), slice(ll1.x, ur1.x)), (slice(ll2.y, ur2.y), slice(ll2.x, ur2.x))

def ind_circle(size, center, radius):
    """Array with points inside a circle set to True otherwise False"""
    x, y = np.mgrid[0:size.y, 0:size.x]
    return ((x - center.y) ** 2 + (y - center.x) ** 2 > radius ** 2) * True

class Control(object):
    def __init__(self, xy, ax1, is_start=False, rotation=0, **kwargs):
        """Class holding the information of a control."""
        self.xy = xy
        self.is_start = is_start
        self.ax1 = ax1
        self.rotation = rotation
        if is_start:
            self.patch1 = RegularPolygon(xy, 3, ** kwargs)
        else:
            self.patch1 = Circle(xy, ** kwargs)
        ax1.add_patch(self.patch1)

    def move(self, xy=None, **kwargs):
        """Move control to position xy"""
        if xy is not None:
            self.xy = xy
        self.ax1.patches.remove(self.patch1)
        if self.is_start:
            self.patch1 = RegularPolygon(self.xy, 3, ** kwargs)
        else:
            self.patch1 = Circle(self.xy, **kwargs)
        self.ax1.add_patch(self.patch1)


class VeryVeryOrienteering(object):
    def __init__(self, ax1, ax2):
        """
        Main class for generating very-very-orienteering maps.
        """
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        self.fig1 = fig1 = ax1.get_figure()
        self.fig2 = fig2 = ax2.get_figure()
        self.tb = plt.get_current_fig_manager().toolbar
        fig1.canvas.mpl_connect('button_press_event', self.onclick)
        fig1.canvas.mpl_connect('button_release_event', self.onrelease)
        fig1.canvas.mpl_connect('pick_event', self.onpick)
        if fig1 != fig2:
            fig2.canvas.mpl_connect('button_press_event', self.onclick)
            fig2.canvas.mpl_connect('button_release_event', self.onrelease)
            fig2.canvas.mpl_connect('pick_event', self.onpick)
        self.mouse_inaxes = None
        self.onclick_time = None
        self.onclick_coords = None
        self.controls = []
        self.lines = []
        self.xy2 = None
        self.radius = 52
        self.window_radius = 150
        self.c_kwargs = dict(fc='none', ec='red', picker=30, radius=50, lw=2, alpha=0.7)
        self.t_kwargs = dict(fc='none', ec='red', picker=30, radius=50, lw=2, alpha=0.7, orientation=pi)
        self.l_kwargs = dict(color='red', picker=5, lw=2, alpha=0.7)
        self.pick_artist = None
        self.pick = 'none'
        self.map2 = self.map = None
        self.imay2 = self.imax = None


    def draw(self, fig= -1):
        """Redraw canvas of the figures of the axes"""
        if self.fig1 == self.fig2 or fig != 2:
            self.fig1.canvas.draw()
        if self.fig1 != self.fig2 and fig != 1:
            self.fig2.canvas.draw()

    def plot_map2(self, very_very=True):
        """Plot second map with rotated controls"""
        if not self.xy2:
            return
        for patch in self.ax2.patches[::-1]:
            self.ax2.patches.remove(patch)
        for line in self.ax2.lines[::-1]:
            self.ax2.lines.remove(line)
        xy2 = self.xy2
        xy = None
        self.map2[:] = 255
        size = Point(*self.map.shape[:2][::-1])
        dif_orientation = 0
        radius = self.window_radius
        for control in self.controls:
            if xy is None:
                control.xy2 = xy2 = self.xy2
            else:
                orientation = slope(xy, control.xy)
                o_ = orientation + dif_orientation
                xy2_new = rad(distance(xy, control.xy), o_) + xy2
                # draw line from xy2 to xy2_new
                line = Line2D(*get_line_data(xy2, xy2_new, self.radius,
                                             self.window_radius * 1.2 if
                                             very_very else self.radius),
                              **self.l_kwargs)
                self.ax2.add_line(line)
                control.xy2 = xy2 = xy2_new
            xy = control.xy
            if very_very:
                dif_orientation += control.rotation
            indb1, indb2 = ind_boxes(size, xy, xy2, radius)
            window = self.map[indb1]
            window = rotate(window, -dif_orientation / pi * 180, axes=(1, 0), reshape=False)
            size_window = Point(*window.shape[:2][::-1])
            indc = ind_circle(size_window, size_window / 2., radius)
            window[indc] = 255
            self.map2[indb2] = window

            if control.is_start:
                self.t_kwargs['orientation'] += control.rotation * very_very
                patch = RegularPolygon(xy2, 3, ** self.t_kwargs)
                self.t_kwargs['orientation'] -= control.rotation * very_very
            else:
                patch = Circle(xy2, ** self.c_kwargs)
            self.ax2.add_patch(patch)
        self.imax2.set_data(self.map2[::self.low_res, ::self.low_res])

    def get_control(self, patch):
        """Return control which corresponds to the given patch"""
        ind = [control.patch1 for control in self.controls].index(patch)
        return self.controls[ind]

    def update_control(self, control):
        """Update control"""
        ind = self.controls.index(control)
        if ind in (0, 1):
            if len(self.controls) > 1:
                self.t_kwargs['orientation'] = slope(self.controls[0].xy,
                                                     self.controls[1].xy)
            self.controls[0].move(**self.t_kwargs)
        if len(self.controls) > 1:
            if ind > 0:
                data = get_line_data(self.controls[ind - 1].xy, control.xy, self.radius)
                self.lines[ind - 1].set_data(data)
            if ind < len(self.controls) - 1:
                data = get_line_data(self.controls[ind + 1].xy, control.xy, self.radius)
                self.lines[ind].set_data(data)

    def create_control(self, xy, rotation=0):
        """Create a new control"""
        if not self.controls:
            self.xy2 = xy
        control = Control(xy, ax1=self.ax1, rotation=rotation,
                          is_start=not self.controls, ** self.c_kwargs)
        self.controls.append(control)
        if len(self.controls) > 1:
            line = Line2D((0, 0), (0, 0), **self.l_kwargs)
            self.lines.append(line)
            self.ax1.add_line(line)
        self.update_control(control)

    def move_control(self, xy, control):
        """
        Move the control to xy
        
        Just calling the move method of control and self.update_control
        """
        control.move(xy, **self.c_kwargs)
        self.update_control(control)

    def move_start_control_map2(self, xy):
        self.xy2 = xy

    def change_controls(self, control1, control2):
        """Exchange control1 with control2"""
        ind1 = self.controls.index(control1)
        ind2 = self.controls.index(control2)
        if ind1 != ind2 != 0:
            xy1 = self.controls[ind1].xy
            self.move_control(self.controls[ind2].xy, self.controls[ind1])
            self.move_control(xy1, self.controls[ind2])

    def split_control(self, xy, line):
        """Insert a new control on the line"""
        ind = self.lines.index(line)
        control = Control(xy, ax1=self.ax1,
                          is_start=not self.controls, ** self.c_kwargs)
        self.controls.insert(ind + 1, control)
        line = Line2D((0, 0), (0, 0), **self.l_kwargs)
        self.lines.insert(ind, line)
        self.ax1.add_line(line)
        self.update_control(control)

    def delete_control(self, control):
        """Delete a control"""
        ind = self.controls.index(control)
        self.ax1.patches.remove(control.patch1)
        self.controls.remove(control)
        if len(self.controls) > 0:
            self.ax1.lines.remove(self.lines.pop(max(0, ind - 1)))
            # other control
            if ind == 0:
                self.controls[0].is_start = True
            self.update_control(self.controls[max(0, ind - 1)])

    def change_orientation(self, line, xy_mouse):
        """Rotate controls"""
        ind = self.ax2.lines.index(line)
        xy1 = self.controls[ind].xy2
        xy2 = self.controls[ind + 1].xy2
        phi1 = slope(xy1, xy2)
        phi2 = slope(xy1, xy_mouse)
        self.controls[ind].rotation += phi2 - phi1


    def onclick(self, event):
        if self.tb.mode != '':
            return
        self.mouse_inaxes = event.inaxes
        self.time_of_onclick = time()

    def onrelease(self, event):
        if self.tb.mode != '' or self.mouse_inaxes != event.inaxes:
            return
        if event.inaxes:
            coords = Point(x=event.xdata, y=event.ydata)
        just_click = time() - self.time_of_onclick < _TIME_CLICK
        first = 'none'
        #first_picked = None
        if self.pick_artist:
            first_picked = self.pick_artist
            first = self.pick
            self.ax1.pick(event)

        if event.inaxes == self.ax1 and just_click and event.button == 1:
            if first == 'none':
                self.create_control(coords)
                self.draw(1)
            elif first == 'line':
                self.split_control(coords, first_picked)
                self.draw(1)
        if event.inaxes == self.ax1 and just_click and event.button == 3 and first == 'control':
                self.delete_control(first_picked)
                self.draw(1)
        if event.inaxes == self.ax1 and not just_click and event.button == 1:
            if first == self.pick == 'control' and  self.pick_artist != first_picked:
                self.change_controls(first_picked, self.pick_artist)
                self.draw(1)
            elif first == 'control':
                self.move_control(coords, first_picked)
                self.draw(1)
        if event.inaxes == self.ax2 and not just_click and event.button == 1:
            if first == 'line':
                self.change_orientation(first_picked, coords)
                self.plot_map2()
                self.draw(2)
            elif first == 'control' and first_picked == self.ax2.patches[0]:
                self.move_start_control_map2(coords)
                self.plot_map2()
                self.draw(2)
        if event.inaxes == self.ax2 and event.button == 3:
            self.plot_map2()
            self.draw(2)
        self.pick = 'none'
        self.pick_artist = None

    def onpick(self, event):
        if self.tb.mode != '':
            return
        self.pick = 'none'
        if isinstance(event.artist, Patch):
            self.pick = 'control'
            try:
                self.pick_artist = self.get_control(event.artist)
            except ValueError:
                self.pick_artist = event.artist
        elif isinstance(event.artist, Line2D):
            self.pick = 'line'
            self.pick_artist = event.artist
        else:
            raise ValueError('Unknown Patch')

    def change_control_radius(self, radius):
        self.c_kwargs['radius'] = self.t_kwargs['radius'] = radius
        self.radius = radius + self.c_kwargs['lw']
        self.l_kwargs['picker'] = self.c_kwargs['picker'] = 0.6 * radius
        self.update_all()
        self.draw()

    def change_window_radius(self, radius):
        self.window_radius = radius
        self.plot_map2()
        self.draw(2)

    def change_lw(self, lw):
        self.c_kwargs['lw'] = self.t_kwargs['lw'] = self.l_kwargs['lw'] = lw
        self.radius = self.c_kwargs['radius'] + lw
        for line in self.lines:
            line.set_lw(lw)
        self.update_all()
        self.draw()

    def change_alpha(self, alpha):
        self.c_kwargs['alpha'] = self.t_kwargs['alpha'] = self.l_kwargs['alpha'] = alpha
        for line in self.lines:
            line.set_alpha(alpha)
        self.update_all()
        self.draw()

    def update_all(self):
        for control in self.controls:
            self.move_control(None, control)
        self.plot_map2()

    def save(self, filename):
        """Save tar file with all relevant maps and pickled data"""
        print('Saving ... this can take a while ...')
        fn_pickle = filename + '_data.pickle'
        img_ext = '.png'
        fn_im1 = filename + '_map' + img_ext
        fn_im2 = filename + '_route' + img_ext
        fn_im3 = filename + '_window' + img_ext
        fn_im4 = filename + '_very' + img_ext
        # Pickle important attributes of instance
        xys = [c.xy for c in self.controls]
        rots = [c.rotation for c in self.controls]
        to_pickle = (xys, self.xy2, rots,
                     self.radius, self.window_radius,
                     self.c_kwargs, self.t_kwargs, self.l_kwargs)
        with open(fn_pickle, 'w') as f:
            pickle.dump(to_pickle, f)
        # Original map
        Image.fromarray(self.map[::-1, :]).save(fn_im1)
        print('1...')
        # Map with route
        extent1 = self.ax1.get_window_extent().transformed(self.fig1.dpi_scale_trans.inverted())
        extent2 = self.ax2.get_window_extent().transformed(self.fig2.dpi_scale_trans.inverted())
        corners = extent1.get_points()
        dpi = self.map.shape[0] / (corners[1, 1] - corners[0, 1])
        self.imax.set_data(self.map)
        self.fig1.savefig(fn_im2, bbox_inches=extent1, dpi=dpi)
        self.imax.set_data(self.map[::self.low_res, ::self.low_res])
        orig_low_res = self.low_res
        self.low_res = 1
        print('2...')
        # Window orienteering map
        self.plot_map2(very_very=False)
        self.fig2.savefig(fn_im3, bbox_inches=extent2, dpi=dpi)
        print('3...')
        # Finally very-very-orienteering map
        self.plot_map2()
        self.fig2.savefig(fn_im4, bbox_inches=extent2, dpi=dpi)
        with tarfile.open(filename + '.tar', 'w') as tar:
            for name in [fn_pickle, fn_im1, fn_im2, fn_im3, fn_im4]:
                tar.add(name)
        os.remove(fn_pickle)
        os.remove(fn_im1)
        os.remove(fn_im2)
        os.remove(fn_im3)
        os.remove(fn_im4)
        self.low_res = orig_low_res
        self.plot_map2()
        print('Finished saving.')

    def load(self, filename, rotate=False):
        """Load image or tar file"""
        if tarfile.is_tarfile(filename):
            with tarfile.open(filename) as tar:
                fn_im = [fn for fn in tar.getnames() if '_map' in fn][0]
                fn_pickle = [fn for fn in tar.getnames() if fn.endswith('.pickle')][0]
                tar.extract(fn_im)
                tar.extract(fn_pickle)
            self.map = np.array(Image.open(fn_im))[::-1, :]
            with open(fn_pickle) as f:
                unpickled = pickle.load(f)
            os.remove(fn_im)
            os.remove(fn_pickle)
        else:
            self.map = np.array(Image.open(filename))[::-1, :]
            if rotate:
                self.map = np.rot90(self.map)
        self.map2 = np.ones_like(self.map) * 255
        self.low_res = max(1, min(self.map.shape[0:2]) // 1000)
        extent = (-0.5, self.map.shape[1] - 0.5, -0.5, self.map.shape[0] - 0.5)
        self.imax = self.ax1.imshow(self.map[::self.low_res, ::self.low_res], aspect='equal',
                                    interpolation='none', origin='lower', extent=extent)
        self.imax2 = self.ax2.imshow(self.map2[::self.low_res, ::self.low_res], aspect='equal',
                                    interpolation='none', origin='lower', extent=extent)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(self.ax1.get_xlim())
        self.ax2.set_ylim(self.ax1.get_ylim())
        if tarfile.is_tarfile(filename):
            (xys, self.xy2, rots,
             self.radius, self.window_radius,
             self.c_kwargs, self.t_kwargs, self.l_kwargs) = unpickled
            for i in range(len(xys)):
                self.create_control(xys[i], rotation=rots[i])
            self.plot_map2()


def main():
    description = __doc__ + """
Here comes a description of arguments and actions which can be performed 
in the session:"""
    epilog = """
Actions which can be performed in window with original map:
   * Create new control by simply clicking
   * Dragging control
   * Insert a control between two controls (by clicking on a line)
   * Exchange two controls by dragging one control over another control
   * Delete control by right-click

Actions which can be performed in the window with the very-very map:
   * Redraw the map by right-click (Map is not automatically redrawn by changes
     in the other window)
   * Move start control by dragging it
   * Rotate controls by dragging the connections   
   
Actions with the sliders:
   * Change size of controls
   * Change line width of controls and lines
   * Change transparency of controls and lines
   * Change radius of windows

Actions with keys:
   * Shift-S (or big S): Save maps and the whole instance in a tar file

Actions with pylab toolbar:
   * Zooming, paning, etc. - just try
    """
    parser = argparse.ArgumentParser(
                        description=description, epilog=epilog,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'file',
        help='Filename of picture or tar file previously saved with Shift-S')
    parser.add_argument(
        '-r', '--rotate', action='store_true',
        help='Rotate picture by 90 degrees (ignored with a tar file)')
    parser.add_argument(
        '-l', '--landscape', action='store_true',
        help='Axes will be arranged upon each other and not next to each other')
    args = parser.parse_args()
    fig = plt.figure()
    ax1 = fig.add_subplot(1 + args.landscape, 1 + (not args.landscape), 1)
    ax2 = fig.add_subplot(1 + args.landscape, 1 + (not args.landscape), 2)
    vvo = VeryVeryOrienteering(ax1, ax2)
    vvo.load(args.file, rotate=args.rotate)
    # Create sliders and connnect the to vvo methods
    axcolor = 'lightgoldenrodyellow'
    axradius = fig.add_axes([0.2, 0.06, 0.35, 0.01], axisbg=axcolor)
    axwindow = fig.add_axes([0.2, 0.03, 0.35, 0.01], axisbg=axcolor)
    axlw = fig.add_axes([0.7, 0.06, 0.15, 0.01], axisbg=axcolor)
    axalpha = fig.add_axes([0.7, 0.03, 0.15, 0.01], axisbg=axcolor)
    sradius = Slider(axradius, 'Radius', 5, 200, valinit=50)
    swindow = Slider(axwindow, 'Window', 5, 500, valinit=150)
    slw = Slider(axlw, 'Lw', 0.5, 5.0, valinit=2.0)
    salpha = Slider(axalpha, 'Alpha', 0.0, 1.0, valinit=0.7)
    sradius.on_changed(vvo.change_control_radius)
    swindow.on_changed(vvo.change_window_radius)
    slw.on_changed(vvo.change_lw)
    salpha.on_changed(vvo.change_alpha)
    def onkey(event):
        if event.key == 'S':
            filename, _ = os.path.splitext(args.file)
            vvo.save(filename)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

if __name__ == '__main__':
    main()
