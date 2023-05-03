# LaplaceSpectrumEllipse

# -*- coding: utf-8 -*-

#######################################################################
#
#  We use 
#             pygmsh 
#  to generate grids, and 
#             scikit-fem  (skfem) 
#  as a finite elements package.
#  Particulary scikit-fem has several dependencies, 
#  which need to be resolved carefully (did not work out
#  automatically under arch-linux/manjaro, but was doable).
#
#  If you use anaconda, please check the version of scikit-fem, 
#  under arch-linux/majaro it was hopeless outdated.
#
######################################################################


################################################
#
# import the necesary objects/packages/methods
#
################################################

# grid generation
import pygmsh



# finite element methods
import skfem as fem
from skfem.models.poisson import laplace, unit_load, mass
from skfem.visuals.matplotlib import plot

# maths
import numpy as np
from scipy import sparse
from math import pi

# mesh plots
import pyvista

################################################
#
# generate mesh by pygmsh, import to skfem
#
################################################

def _sanitize(points, cells):
    # return points and cells in a way we'll need later
    # (did just copy-and-paste this routine, no idea what happens here :( ))
    uvertices, uidx = np.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = points[uvertices]
    return points, cells


### create mesh with a given boundary (polygon) ###

# produce a polygon for ellipse
a  = 2 #in x direction
b = 4 #in y direction
n = 80; #anz Stützstellen
stuetzstellen = []
def getRadius(phi,a,b):
    eps = 0.01
    radius = (a*b) / np.sqrt((b**2*np.cos(phi)**2) + (a**2*np.sin(phi)**2))
    perturbed = radius * (1+eps*np.cos(phi*16)) #2 pi periodic, high frequency, around 1, low amplitude
    return radius
    #return perturbed

for i in range(n):
    x = getRadius(2*pi*i/n, a, b)*np.cos(2*pi*i/n)  
    y = getRadius(2*pi*i/n, a, b)*np.sin(2*pi*i/n)
    stuetzstellen.append([x,y,0])

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(stuetzstellen, mesh_size = 0.1)
    mesh = geom.generate_mesh()


#plot the mesh
pyvista.set_plot_theme("document")
p = pyvista.Plotter(window_size=(800, 800))
p.add_mesh(
    mesh=pyvista.from_meshio(mesh),
    show_edges=True,
)
p.view_xy()
p.show()

### convert into a mesh that can be used by skfem ###

# extract points and triangles 
tri = mesh.get_cells_type("triangle")
assert len(tri) > 0
points, cells = _sanitize(mesh.points, tri)

# reformat points and triangles (data type, and transpose/not transpose)
p = np.array([points[:,0],  points[:,1]], dtype=np.float64)
t = np.array(cells, dtype=np.float64).T

# create mesh (we could refine the mesh slightly 
# using fem.MeshTri(p,t).refined(2), 
# but perhaps better to control the grid in pygmsh)
myMesh = fem.MeshTri(p,t)


###############################################
#
# setup Poisson equation 
#            -\Delta u = 1
# (just for fun, and to see that it works out :) )
#
###############################################

# piecewise linear finite elements
e = fem.ElementTriP1(); 

# create mapping for the finite element approximation 
basis = fem.CellBasis(myMesh, e)

# create matrix for Laplace
A = laplace.assemble(basis)

# create vector for "1" (Poisson equation, -\Delta u = 1) 
b = unit_load.assemble(basis)

# enforce Dirichlet boundary conditions
A, b = fem.enforce(A, b, D=myMesh.boundary_nodes())

# solve the equation 
x = fem.solve(A,b);

# visualize :)
#plot(myMesh, x, shading='gouraud', colorbar=True).show()




###############################################
#
# solve Eigenvalue problem
# (we need here the finite elements basis and 
#  the matrix A produced above)
#
###############################################


# we need that matrix - but why?
M = fem.asm(mass, basis);

# compute the first k=20 eigenvalues, and determine the eigenvektors along
k = 4
eVals, eVectors = fem.solve(*fem.condense(A, M, D=basis.get_dofs()), 
                            solver=fem.solver_eigen_scipy_sym(k=k)   )

# plot the eigenfunction (using the real part;
# the eigenvalue/eigenfunctions are real anyway)
for ii in range(k):
    plot(myMesh, eVectors[:,ii].real, shading='gouraud', colorbar=True).show()