from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import openmc
import math
import pandas as pd
import time
from tabulate import tabulate
from collections import OrderedDict

K_TARG = 1.0
TOL = 0.0001
FINAL_UNC = 0.0005


def create_mats(mat_spec):
    
    """Create materials for study and returns the materials.
    """

    # Define the material
    iso = mat_spec.name
    dens = mat_spec.dens

    # Instantiate Nuclides
    candidate = openmc.Nuclide(iso)
    he3 = openmc.Nuclide('He3')

    # Create materials
    fuel = openmc.Material(name=iso)
    fuel.set_density('g/cm3', dens)
    fuel.add_nuclide(candidate, 1.0)

    air = openmc.Material(name='air')
    air.set_density('g/cm3', 0.001225)
    air.add_nuclide(he3, 1.0)

    materials_file = openmc.Materials((fuel, air))
    materials_file.default_xs = '71c'
    materials_file.export_to_xml()

    return fuel, air

def create_tally(radius):

    """Create tally mesh???
    """

    tallies_file = openmc.Tallies()

    mesh = openmc.Mesh()
    mesh.dimension = [100, 100]
    mesh.lower_left = [-radius, -radius]
    mesh.upper_right = [radius, radius]

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_filter.mesh = mesh

    tally = openmc.Tally(name='flux')
    tally.filters = [mesh_filter]
    tally.scores = ['flux', 'fission']
    tallies_file.append(tally)
    tallies_file.export_to_xml()

def create_settings_file(particles):

    """creates settings file with input numer of particles.
    """

    settings_file = openmc.Settings()
    settings_file.verbosity = 1
    settings_file.batches = 100
    settings_file.inactive = 10
    settings_file.particles = particles

    return settings_file

def create_geom(radius, fuel, air):

    """Create geometry for some radius sphere
    """

    # Create geometry
    sphere = openmc.Sphere(R=radius)
    min_x = openmc.XPlane(x0=-radius, boundary_type='vacuum') 
    max_x = openmc.XPlane(x0=+radius, boundary_type='vacuum') 
    min_y = openmc.YPlane(y0=-radius, boundary_type='vacuum') 
    max_y = openmc.YPlane(y0=+radius, boundary_type='vacuum') 
    min_z = openmc.ZPlane(z0=-radius, boundary_type='vacuum')
    max_z = openmc.ZPlane(z0=+radius, boundary_type='vacuum') 

    # Create Universe
    universe = openmc.Universe(name='Universe')

    fuel_cell = openmc.Cell(name='fuel')
    fuel_cell.fill = fuel
    fuel_cell.region = -sphere
    universe.add_cell(fuel_cell)

    air_cell = openmc.Cell(name='air')
    air_cell.fill = None
    air_cell.region = +sphere
    universe.add_cell(air_cell)

    # Create root cell
    root_cell = openmc.Cell(name='root_cell')
    root_cell.fill = universe
    root_cell.region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
    root_universe = openmc.Universe(universe_id=0, name='root universe')
    root_universe.add_cell(root_cell)

    geometry = openmc.Geometry()
    geometry.root_universe = root_universe
    geometry.export_to_xml()

def create_source(radius, settings_file):
    """Create the source distribution.
    """

    bounds = [-radius, -radius, -radius, radius, radius, radius]
    uniform_dist = \
        openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
    settings_file.source = openmc.source.Source(space=uniform_dist)
    settings_file.export_to_xml()

def run_calc(radius, fuel, air, particles):

    """Runs the computation and returns the k-calc.
    """

    settings_file = create_settings_file(particles)
    create_geom(radius, fuel, air)
    create_source(radius, settings_file)

    create_tally(radius)
    openmc.run() 
    sp = openmc.StatePoint('statepoint.100.h5')

    return sp.k_combined[0], sp.k_combined[1]

def compute_critical_radius(mat_spec, method='false-position'):

    """Compute critical radius using root finding algorithm.
    """

    # Setup materials.
    fuel, air = create_mats(mat_spec)

    # Setup initial convergence critiera and particles per batch.
    particles = 4000
    tighten_criteria = TOL*20
    iters = 2

    # Run initial guesses
    r1 = 0.1
    r2= 20.0
    print(" ~-~-~-~ {} ~-~-~-~-~".format(fuel.name))
    print("Beginning initial guesses using {} cm and {} cm".format(r1, r2))
    f1 = run_calc(r1, fuel, air, particles)[0] - K_TARG
    f2 = run_calc(r2, fuel, air, particles)[0] - K_TARG

    # If outside r_2, consider it a non starter.
    if f2 < 0:
        print("{} is a dud.".format(fuel.name))
        return 'N/A', 'N/A', 'N/A', 'N/A'

    # Using search algorithm, find critical radius.
    while True:

        # Calculate next next guess radius and compute k-eff.
        r_new = next_guess(r1, r2, f1, f2, method)
        k_new, unc = run_calc(r_new, fuel, air, particles)
        f_new = k_new - K_TARG
        print("Iteration: {}    K-eff: {}    Radius: {}".
              format(iters, k_new, r_new))

        # Check if conv criteria should be tightened.
        if (r2 - r1)/2 < tighten_criteria:
            # Updating particles so that uncertainty is within conv criteria.
            tighten_criteria = 0
            particles = int(pow(unc/FINAL_UNC, 2)*particles)
            print("Narrowing Convergence Criteria and Updating # Particles")
            print(" - Desired Uncertainty in k-eff {}".format(FINAL_UNC))
            print(" - Uncertainty: {}".format(unc))
            print(" - Updating Particles: {}".format(particles))

        # Determine if converged
        if f_new == 0 or (r2 - r1)/2 < TOL:
            print("Critical Radius Calculated: {}\n\n".format(r_new))
            return r_new, iters, k_new, unc

        # Update boundaries
        r1, r2, f1, f2 = redefine_boundaries(r1, r2, f1, f2, f_new, r_new,
                                             method)
        iters += 1

def next_guess(r1, r2, f1, f2, method):

    """Compute next guess in root finding algorithm.
    """

    if method == 'false-position':
        return (f1*r2 - f2*r1)/(f1 - f2)
    elif method == 'bisection':
        return (r2 - r1)/2 + r1
    else:
        print('Invalid method requested')
        raise Exception

def redefine_boundaries(r1, r2, f1, f2, f_new, r_new, method):

    """Shrink boundaries for closed method algorithms.
    """

    if method =='false-position':
        if f_new * f2 > 0:
            r2 = r_new
            f2 = f_new
        elif f1 * f_new > 0:
            r1 = r_new
            f1 = f_new

    elif method == 'bisection':
        if f_new < 0:
            r1 = r_new
        else:
            r2 = r_new

    return r1, r2, f1, f2

#####################################################
#####################################################
#####################################################
#####################################################

class Material():
    def __init__(self, name, dens):
        self.name = name
        self.dens = dens

data = OrderedDict([('Isotope', []), ('Radius (cm)', []),
                    ('Mass (kg)', []), ('Iters', []), ('Method', []),
                    ('k-eff', []), ('Unc (abs)', []), ('Time (s)', [])])

balls = []
balls.append(Material('U238', 19.1))
balls.append(Material('U235', 19.1))
balls.append(Material('U234', 19.1))
balls.append(Material('U233', 19.1))


# Append run information
for ball in balls:
    for meth in ['false-position', 'bisection']:
        # Compute the answer, and time it.
        start = time.time()
        radius, iters, k_eff, unc = compute_critical_radius(ball, meth)
        timed = time.time() - start

        data['Isotope'].append(ball.name)
        data['Method'].append(meth)
        data['Time (s)'].append(timed)

        if isinstance(radius, str):
            mass = 'N/A'
        else:
            mass = ball.dens*4/3000*math.pi*pow(radius,3)

        data['k-eff'].append(k_eff)
        data['Unc (abs)'].append(unc)
        data['Iters'].append(iters)
        data['Radius (cm)'].append(radius)
        data['Mass (kg)'].append(mass)



df = pd.DataFrame(data)

print tabulate(df, headers='keys', tablefmt='psql', showindex=False,
               stralign='center', numalign='center')
