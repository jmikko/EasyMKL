"""
==========
KOMD GUI
==========

A simple graphical frontend for Libsvm mainly intended for didactic
purposes. You can create data points by point and click and visualize
the decision region induced by different kernels and parameter settings.

To create positive examples click the left mouse button; to create
negative examples click the right button.

"""
from __future__ import division

print __doc__

# Author: Peter Prettenhoer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet

import Tkinter as Tk
import sys
import numpy as np

from sklearn import svm
import komd
from sklearn.datasets import dump_svmlight_file

y_min, y_max = -50, 50
x_min, x_max = -50, 50

from cvxopt import matrix
from scipy.spatial import ConvexHull
from weakref import ref

ch1,ch2=0,0
has_margin = False
has_margin2 = False
t1,t2 = [],[]
bias = 0
gamma = []
isFitted = False
neg_c,pos_c = False,False

class Model(object):
    """The Model which hold the data. It implements the
    observable in the observer pattern and notifies the
    registered observers on change event.
    """

    def __init__(self):
        self.observers = []
        self.surface = None
        self.data = []
        self.cls = None
        self.surface_type = 0

    def changed(self, event):
        """Notify the observers. """
        for observer in self.observers:
            observer.update(event, self)

    def add_observer(self, observer):
        """Register an observer. """
        self.observers.append(observer)

    def set_surface(self, surface):
        self.surface = surface

    def dump_svmlight_file(self, file):
        data = np.array(self.data)
        X = data[:, 0:2]
        y = data[:, 2]
        dump_svmlight_file(X, y, file)


class Controller(object):
    def __init__(self, model):
        self.model = model
        self.kernel = Tk.IntVar()
        self.surface_type = Tk.IntVar()
        # Whether or not a model has been fitted
        self.fitted = False

    def fit(self):
        global isFitted
        isFitted = True
        print "fit the model"
        train = np.array(self.model.data)
        X = train[:, 0:2]
        y = train[:, 2]

        lam = float(self.complexity.get())
        gamma = float(self.gamma.get())
        coef0 = float(self.coef0.get())
        degree = int(self.degree.get())
        kernel_map = {0: "linear", 1: "rbf", 2: "poly"}
        #if len(np.unique(y)) == 1:
        #    clf = svm.OneClassSVM(kernel=kernel_map[self.kernel.get()],
        #              gamma=gamma, coef0=coef0, degree=degree)
        #    clf.fit(X)
        #else:
        #mysvm = svm.SVC(kernel=kernel_map[self.kernel.get()], C=1000,
        #                  gamma=gamma, coef0=coef0, degree=degree)
        #mysvm.fit(X, y)
        #l = 0.1;
        clf = komd.KOMD(lam=lam, Kf=kernel_map[self.kernel.get()], rbf_gamma=gamma, poly_deg=degree, poly_coeff=coef0)

        clf.fit(X,y)
        #print clf.gamma
        #global gamma, bias
        #gamma = clf.gamma
        #bias = clf.bias
	
        if hasattr(clf, 'score'):
            print "Accuracy:", clf.score(X, y) * 100
        X1, X2, Z = self.decision_surface(clf)
        self.model.clf = clf
        #self.model.svm = mysvm
        self.clf = clf
        #self.mysvm = mysvm
        self.model.set_surface((X1, X2, Z))
        self.model.surface_type = self.surface_type.get()
        self.fitted = True
        self.model.changed("surface")

    def decision_surface(self, cls):
	delta = 1
        x = np.arange(x_min, x_max + delta, delta)
        y = np.arange(y_min, y_max + delta, delta)
        X1, X2 = np.meshgrid(x, y)
        Z = cls.decision_function(np.c_[X1.ravel(), X2.ravel()])
        Z = np.array(Z)#random stuff
        Z = Z.reshape(X1.shape)
        return X1, X2, Z

    def clear_data(self):
        self.model.data = []
        self.fitted = False
        self.model.changed("clear")

    def add_example(self, x, y, label):
        self.model.data.append((x, y, label))
        #self.refit()
        self.model.changed("example_added")
        # update decision surface if already fitted.
        self.refit()
        #self.model.changed("distance")

    def refit(self):
        """Refit the model if already fitted. """
        if self.fitted:
            self.fit()
            self.model.changed("distance")


class View(object):
    """Test docstring. """
    def __init__(self, root, controller):
        f = Figure()
        nticks = 10
        ax = f.add_subplot(111, aspect='1')
    	ax.set_xticks([x*(x_max-x_min)/nticks+x_min for x in range(nticks+1)])
    	ax.set_yticks([y*(y_max-y_min)/nticks+y_min for y in range(1,nticks+1)])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        self.controllbar = ControllBar(root, controller)
        self.f = f
        self.ax = ax
        self.canvas = canvas
        self.controller = controller
        self.contours = []
        self.c_labels = None
        self.plot_kernels()
        
    def set_dim(self):
        nticks = 10
        self.ax.set_xticks([x*(x_max-x_min)/nticks+x_min for x in range(nticks+1)])
        self.ax.set_yticks([y*(y_max-y_min)/nticks+y_min for y in range(1,nticks+1)])

    def plot_kernels(self):
        self.ax.text(-50, -60, "Linear: $u^T v$")
        self.ax.text(-20, -60, "RBF: $\exp (-\gamma \| u-v \|^2)$")
        self.ax.text(10, -60, "Poly: $(\gamma \, u^T v + r)^d$")

    def onclick(self, event):
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_example(event.xdata, event.ydata, 1)
            elif event.button == 3:
                self.controller.add_example(event.xdata, event.ydata, -1)

    def update_example(self, model, idx):
        x, y, l = model.data[idx]
        if l == 1:
            color = 'w'
        elif l == -1:
            color = 'k'
        self.ax.plot([x], [y], "%so" % color, scalex=0.0, scaley=0.0)
        global neg_c,pos_c
        if l==1:
            pos_c = True
        if l!=1:
            neg_c = True
        #print neg_c,pos_c
        
    def draw_distance(self, model):
        global has_margin
        train = np.array(model.data)
        gamma = model.clf.gamma
        #print len(gamma)," ",len(train)
        #cx1 = sum([train[i,0]*gamma[i] for i in range(len(train)) if train[i,2]==1])
        #cy1 = sum([train[i,1]*gamma[i] for i in range(len(train)) if train[i,2]==1])
        #cx2 = sum([train[i,0] for i in range(len(train)) if train[i,2]!=1])
        #cy2 = sum([train[i,1] for i in range(len(train)) if train[i,2]!=1])
        #t1 = [example for example in enumerate(train) if example[2==1]]
        t1,t2=[],[]
        cx1,cy1,cx2,cy2=0,0,0,0
        gam1, gam2 = 0,0
        for i,example in enumerate(train):
            if example[2]==1:
	            t1.append(example[0:2])
	            cx1 += example[0]*gamma[i]
	            cy1 += example[1]*gamma[i]
	            gam1 += gamma[i]
            else:
    	        t2.append(example[0:2])
    	        cx2 += example[0]*gamma[i]
    	        cy2 += example[1]*gamma[i]
    	        gam2 += gamma[i]
    	if len(t1)>0:
            cx1 /= gam1
            cy1 /= gam1
        if len(t2)>0:
            cx2 /= gam2
            cy2 /= gam2
        #if (has_margin):
        #    self.ax.lines.pop(len(t1)+len(t2)-1)
        global has_margin,ch1,ch2
        #if (has_margin):
        #    print len(t1)+len(t2)+ch1+ch2-1
        #    self.ax.lines.pop(len(t1)+len(t2)+ch1+ch2-1)
            #self.ax.lines.pop(1)
        #    has_margin = False
        if len(t2)>0 and len(t1)>0:
            has_margin = True
            self.ax.plot([cx1,cx2],[cy1,cy2], '--', color='black')
            
        

    def convex_hull(self, model):
        train = np.array(model.data)
        global t1,t2
        t1,t2 = [],[]
        
        for i,example in enumerate(train):
            if example[2]==1:
	            t1.append(example[0:2])
            else:
    	        t2.append(example[0:2])
        global ch1,ch2,has_margin
        if (has_margin):
            self.ax.lines.pop(len(t1)+len(t2)-1)
        for i in range(ch1+ch2):
            self.ax.lines.pop(len(t1)+len(t2)-1)
        if len(t1) > 2:
            X1 = np.matrix(np.array(t1))
            hull1 = ConvexHull(X1);    
            for simplex in hull1.simplices:
                self.ax.plot(X1[simplex,0],X1[simplex,1], color='grey',linewidth=1)
            ch1 = len(hull1.simplices)
            
        if len(t2) > 2:
            X2 = np.matrix(np.array(t2))
            hull2 = ConvexHull(X2);
            for simplex in hull2.simplices:
                self.ax.plot(X2[simplex,0],X2[simplex,1], color='0.25')
            #self.ax.plot(cx2,cy2,'+',ms=10, color="blue")
            ch2 = len(hull2.simplices)
        global isFitted
        #print len(train),"\t"
        #if isFitted:
        #    self.draw_distance(model,model.clf.gamma)
        self.set_dim()
        
        
        
    def update(self, event, model):
        global neg_c,pos_c
        
        if event == "examples_loaded":
            for i in xrange(len(model.data)):
                self.update_example(model, i)
            self.convex_hull(model)

        if event == "example_added":
            self.update_example(model, -1)
            #self.update("clear",model)
            #self.update("examples_loaded",model)
            self.convex_hull(model)
            if neg_c and pos_c:
                self.controller.fit()
                self.draw_distance(model)
                
        if event == "distance1":
            self.convex_hull(model)
            if neg_c and pos_c:
                self.controller.fit()
                self.draw_distance(model)

        if event == "clear":
            self.ax.clear()
            self.contours = []
            #self.c_labels = None
            global ch1,ch2,t1,t2,has_margin
            #for i in range(ch1+ch2):
            #    self.ax.lines.pop(i)
            ch1,ch2 = 0,0
            #t1 = t2 = [],[]
            has_margin = False
            neg_c,pos_c = False,False
            self.plot_kernels()
            self.set_dim()

        if event == "surface":
            self.remove_surface()
            #self.plot_support_vectors(model.svm.support_vectors_)
            self.plot_decision_surface(model.surface, model.surface_type)
            #self.draw_distance(model,model.clf.gamma)   
        self.canvas.draw()

    def remove_surface(self):
        """Remove old decision surface."""
        if len(self.contours) > 0:
            for contour in self.contours:
                if isinstance(contour, ContourSet):
                    for lineset in contour.collections:
                        lineset.remove()
                else:
                    contour.remove()
            self.contours = []

    def plot_support_vectors(self, support_vectors):
        """Plot the support vectors by placing circles over the
        corresponding data points and adds the circle collection
        to the contours list."""
        cs = self.ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                             s=80, edgecolors="k", facecolors="none")
        self.contours.append(cs)

    def plot_decision_surface(self, surface, type):
        X1, X2, Z = surface
        if type == 0:
            levels = [-1.0, 0.0, 1.0]
            linestyles = ['dashed', 'solid', 'dashed']
            colors = 'k'
            self.contours.append(self.ax.contour(X1, X2, Z, levels,
                                                 colors=colors,
                                                 linestyles=linestyles))
            #global gamma,bias
            #Z = Z*bias
            #self.contours.append(self.ax.contour(X1.T, -X2.T, -Z, [0],
            #                                     colors=colors,
            #                                     linestyles=['dashed']))
        elif type == 1:
            self.contours.append(self.ax.contourf(X1, X2, Z, 10,
                                             cmap=matplotlib.cm.bone,
                                             origin='lower',
                                             alpha=0.85))
            self.contours.append(self.ax.contour(X1, X2, Z, [0.0],
                                                 colors='k',
                                                 linestyles=['solid']))
        else:
            raise ValueError("surface type unknown")


class ControllBar(object):
    def __init__(self, root, controller):
        fm = Tk.Frame(root)
        kernel_group = Tk.Frame(fm)
        Tk.Radiobutton(kernel_group, text="Linear", variable=controller.kernel,
                       value=0, command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="RBF", variable=controller.kernel,
                       value=1, command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="Poly", variable=controller.kernel,
                       value=2, command=controller.refit).pack(anchor=Tk.W)
        kernel_group.pack(side=Tk.LEFT)

        valbox = Tk.Frame(fm)
        controller.complexity = Tk.StringVar()
        controller.complexity.set("1.0")
        lam = Tk.Frame(valbox)
        Tk.Label(lam, text="lambda:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(lam, width=6, textvariable=controller.complexity).pack(
            side=Tk.LEFT)
        lam	.pack()

        controller.gamma = Tk.StringVar()
        controller.gamma.set("0.01")
        g = Tk.Frame(valbox)
        Tk.Label(g, text="gamma:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(g, width=6, textvariable=controller.gamma).pack(side=Tk.LEFT)
        g.pack()

        controller.degree = Tk.StringVar()
        controller.degree.set("3")
        d = Tk.Frame(valbox)
        Tk.Label(d, text="degree:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(d, width=6, textvariable=controller.degree).pack(side=Tk.LEFT)
        d.pack()

        controller.coef0 = Tk.StringVar()
        controller.coef0.set("0")
        r = Tk.Frame(valbox)
        Tk.Label(r, text="coef0:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(r, width=6, textvariable=controller.coef0).pack(side=Tk.LEFT)
        r.pack()
        valbox.pack(side=Tk.LEFT)

        cmap_group = Tk.Frame(fm)
        Tk.Radiobutton(cmap_group, text="Hyperplanes",
                       variable=controller.surface_type, value=0,
                       command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(cmap_group, text="Surface",
                       variable=controller.surface_type, value=1,
                       command=controller.refit).pack(anchor=Tk.W)

        cmap_group.pack(side=Tk.LEFT)

        train_button = Tk.Button(fm, text='Fit', width=5,
                                 command=controller.fit)
        train_button.pack()
        fm.pack(side=Tk.LEFT)
        Tk.Button(fm, text='Clear', width=5,
                  command=controller.clear_data).pack(side=Tk.LEFT)


def get_parser():
    from optparse import OptionParser
    op = OptionParser()
    op.add_option("--output",
              action="store", type="str", dest="output",
              help="Path where to dump data.")
    return op


def main(argv):
    op = get_parser()
    opts, args = op.parse_args(argv[1:])
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("KOMD GUI")
    view = View(root, controller)
    model.add_observer(view)
    Tk.mainloop()

    if opts.output:
        model.dump_svmlight_file(opts.output)

if __name__ == "__main__":
    main(sys.argv)
