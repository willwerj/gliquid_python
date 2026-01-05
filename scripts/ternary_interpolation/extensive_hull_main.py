'''
Author: Abrar Rauf

Description: This script contains several methods related to constructing lower convex hulls in all extensive space
and extracting the hyperplane equations and inclinations of the simplices of the lower hull
'''

from scipy.spatial import ConvexHull
import numpy as np 
import sympy as smp
import json
import os
import math
import warnings
import time

def gen_hsx_lowerhull(points):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    # Input: points = array of coordinates of the points in the Xi-S-H space
    # Output: simplices = array of simplices that form the lower convex hull of the Xi-S-H space

    # determine the dimensionality of the points
    dim = points.shape[1]

    # initialize bounds for Xi
    x_list = []
    for i in range(dim-1):
        sublist = []
        for j in range(dim-2):
            if i == 0:
                sublist.append(0)
            else:
                if j == i-1:
                    sublist.append(1)
                else:
                    sublist.append(0)
            
        x_list.append(sublist)

    # initialize bounds for S and H
    s_index = dim-2
    s_data = points[:, s_index] 
    s_min = np.min(s_data) 
    s_max = np.max(s_data)

    h_index = dim-1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    upper_bound = 4*h_max

    # create a list of fictitious points
    fict_coords = []
    for i in range(dim-1):
        fict_coord = []
        for j in range(dim-2):
            fict_coord.append(x_list[i][j])
        fict_coord.append(s_min)
        fict_coord.append(upper_bound)
        fict_coords.append(fict_coord)

    for i in range(dim-1):
        fict_coord = []
        for j in range(dim-2):
            fict_coord.append(x_list[i][j])
        fict_coord.append(s_max)
        fict_coord.append(upper_bound)
        fict_coords.append(fict_coord)

    fict_coords = np.array(fict_coords)

    # add the fictitious points to the original distribution
    fict_points = np.array(fict_coords)
    new_points = np.vstack((points, fict_points))

    # take the total convex hull
    new_hull = ConvexHull(new_points)

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull = []
    def check_common_rows(arr1, arr2):
        return any((arr1 == row).all(axis=1).any() for row in arr2)
    
    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fict_points):
            lower_hull.append(simplex)

    arr_lowerhull = np.array(lower_hull)

    return arr_lowerhull
    



def hsx_hyperplane_eqns(points, lower_hull, multiplier, partial_indices):
    # Function to calculate hyperplane equations and partial derivatives of lower hull simplices
    # Input: points = array of coordinates of the points in the Xi-S-H space
    #        lower_hull = array of simplices that form the lower convex hull of the Xi-S-H space
    #        multiplier = multiplier to correct scaled enthalpy values
    # output: all_partial_derivatives = list of all the partial derivatives of the hyperplane equations of the simplices

    # determine the dimensionality of the points
    dim = points.shape[1]
    
    # extract the vertices of each simplex of the lower hull
    all_vertices = []
    for simplex in lower_hull:
        vertices = points[simplex]
        all_vertices.append(vertices)
    all_vertices = np.array(all_vertices)

    
    # initialize array of symbolic variables that form the basis of the coordinate space
    x = []
    for i in range(dim):
        x.append(smp.Symbol('x{}'.format(i)))
    x = np.array(x)
    x = np.transpose(x)

    # initialize array of symbolic variables that represent the coordinates of the vertices
    A = []
    for i in range(dim-1):
        a = []
        for j in range(dim):
            a.append(smp.Symbol('a{}{}'.format(j, i)))
        A.append(a)
    A = np.array(A)
    A = np.transpose(A)

    # intialize a symbolic vector that points to one of the vertices of the simplex
    c = []
    for i in range(dim):
        c.append(smp.Symbol('c{}'.format(i)))
    c = np.array(c)
    c = np.transpose(c)
  
    # horizontally stack the A and x matrices
    M = np.hstack((A, x[:, np.newaxis]))

    # convert the matrix to a sympy matrix
    final_M = smp.Matrix(M)
    
    # analytically solve for the general symbolic equation of the hyperplane 
    sym_normal_vec = smp.det(final_M).simplify()

    # extract the coefficients of the normal vector and put them in an array
    normal_vec = []
    for i in range(dim):
        normal_vec.append(sym_normal_vec.coeff('x{}'.format(i)))
    normal_vec = np.array(normal_vec)

    # symbolically calculate the dot product of the normal vector and the coordinate vector
    intercept = -np.dot(normal_vec, c)
    
    hyperplane_eqn = smp.Add(sym_normal_vec, intercept)

    # express the equation in terms of x_dim
    solution_form = smp.solve(hyperplane_eqn, 'x{}'.format(dim-1))

    # analytically solve for the partial derivatives of the hyperplane equation
    partial_formulae = []

    for i in range(dim-1):
        partial_formulae.append(smp.diff(solution_form[0], 'x{}'.format(i)))

    # create a new list of partial formulae that only contains the elements from the list of partial_indices
    partial_formulae = [partial_formulae[i] for i in partial_indices]
    print("partial derivatives: ", partial_formulae)

    # initialize an empty list to store all the to be computed partial derivatives
    all_partial_derivatives = []

    # iterate through the vertices of each simplex
    for vertices in all_vertices:

        # create edge vectors by subtracting the first coordinate of each vertex from the remaining coordinates
        vector_list = []
        for i in range(dim-1):
            vector = np.transpose(vertices[i+1] - vertices[0])
            vector = vector.round(5)
            vector = smp.Matrix(vector)
            vector_list.append(vector)


        # create a matrix where the vectors are stored as columns
        def create_matrix(vectors):
            matrix = np.column_stack(vectors)
            return matrix
        
        matrix = create_matrix(vector_list)
        
        # function that takes in a 2D matrix reads each element column wise and returns a list of all the elements
        def read_matrix(matrix):
            elements = []
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[0]):
                    elements.append(matrix[j, i])
            return elements
        
        # convert symbolic and numeric matrices to lists
        var_ele = read_matrix(A)
        mat_ele = read_matrix(matrix)

        # initialize an empty dictionary to map the symbolic variables to the numeric values
        var_dict = {}
        for key, value in zip(var_ele, mat_ele):
            var_dict[key] = value

        coord_dict = {}
        for key, value in zip(c, vertices[0]):
            coord_dict[key] = value

        # concatenate var_dict and coord_dict
        var_dict.update(coord_dict)

        
        # substitute the symbolic variables with the numeric values to evaluate the partial derivatives
        partials = []
        for form in partial_formulae:
            # if TypeError is raised, move on to the next iteration while appending a 'nan' to the list of partials
            try:
                eqn = form.subs(var_dict) 
                val = smp.N(eqn)
                val = float(val)
                val = val*multiplier
                # take real part of the value
                partials.append(val)
            except TypeError:
                partials.append('nan')
                continue 
        
        # append  partial derivatives to the list of all  partial derivatives
        all_partial_derivatives.append(partials)

    
    return all_partial_derivatives

def gen_hyperplane_eqns(points, lower_hull = [], direct_vertices = False, multiplier = 1, partial_indices = [0], evaluate_eqns=False, evaluate_partials=True):
    
    # print("points: ")
    # print(points)
    # determine the dimensionality of the points
    dim = points.shape[1]
    
    print(dim)

    # extract the vertices of each simplex of the lower hull
    if direct_vertices:
        all_vertices = np.array([points])

    else:
        all_vertices = []
        for simplex in lower_hull:
            vertices = points[simplex]
            all_vertices.append(vertices)
        all_vertices = np.array(all_vertices)
        # print("all vertices: ")
        # print(all_vertices)

    # initialize array of symbolic variables that form the basis of the coordinate space
    x = []
    for i in range(dim):
        x.append(smp.Symbol('x{}'.format(i)))
    x = np.array(x)
    x = np.transpose(x)

    print(x)

    # initialize array of symbolic variables that represent the coordinates of the vertices
    A = []
    for i in range(dim-1):
        a = []
        for j in range(dim):
            a.append(smp.Symbol('a{}{}'.format(j, i)))
        A.append(a)
    A = np.array(A)
    A = np.transpose(A)

    # intialize a symbolic vector that points to one of the vertices of the simplex
    c = []
    for i in range(dim):
        c.append(smp.Symbol('c{}'.format(i)))
    c = np.array(c)
    c = np.transpose(c)
  
    # horizontally stack the A and x matrices
    M = np.hstack((A, x[:, np.newaxis]))

    # convert the matrix to a sympy matrix
    final_M = smp.Matrix(M)
    
    # analytically solve for the general symbolic equation of the hyperplane 
    sym_normal_vec = smp.det(final_M).simplify()

    # extract the coefficients of the normal vector and put them in an array
    normal_vec = []
    for i in range(dim):
        normal_vec.append(sym_normal_vec.coeff('x{}'.format(i)))
    normal_vec = np.array(normal_vec)

    # symbolically calculate the dot product of the normal vector and the coordinate vector
    intercept = -np.dot(normal_vec, c)
    
    hyperplane_eqn = smp.Add(sym_normal_vec, intercept)

    # express the equation in terms of x_dim
    solution_form = smp.solve(hyperplane_eqn, 'x{}'.format(dim-1))

    # analytically solve for the partial derivatives of the hyperplane equation
    partial_formulae = []

    for i in range(dim-1):
        partial_formulae.append(smp.diff(solution_form[0], 'x{}'.format(i)))

    # create a new list of partial formulae that only contains the elements from the list of partial_indices
    partial_formulae = [partial_formulae[i] for i in partial_indices]

    # create an x_list
    x_list = []
    for entry in x:
        x_list.append(entry)

    hyperplane_eqn = smp.collect(hyperplane_eqn, x_list)

    remaining_expr = hyperplane_eqn

    coefficients = []
    for xi in x_list:
        coefficients.append(hyperplane_eqn.coeff(xi))
        remaining_expr = remaining_expr - hyperplane_eqn.coeff(xi)*xi

    hyperplane_coeffs = coefficients + [remaining_expr]

    # conver the hyperplane equation to a string
    hyperplane_eqn = str(hyperplane_coeffs)
    print("hyperplane equation: ", hyperplane_eqn)

    # convert the partial derivatives to strings
    partial_formulae = [str(partial) for partial in partial_formulae]
    print("partial derivatives: ", partial_formulae)

    # caching hyperplane equations
    hyperplane_eqn_dict = {}
    if os.path.exists('hyperplane_eqns.json'):
        with open('matrix_data_jsons/hyperplane_eqns.json', 'r') as f:
            hyperplane_eqn_dict = json.load(f)

    # check if the dim is already in the dict
    if str(dim) not in hyperplane_eqn_dict:
        hyperplane_eqn_dict[str(dim)] = hyperplane_eqn

        # with open('matrix_data_jsons/hyperplane_eqns.json', 'w') as f:
        #     json.dump(hyperplane_eqn_dict, f)

    # caching partial derivatives
    partial_derivatives_dict = {}
    if os.path.exists('partial_derivatives.json'):
        with open('partial_derivatives.json', 'r') as f:
            partial_derivatives_dict = json.load(f)

    else:
        # create an empty partial derivatives.json file
        raise FileNotFoundError("partial_derivatives.json file not found. Please create the file before running this function.")

    if str(dim) not in partial_derivatives_dict:
        partial_derivatives_dict[str(dim)] = {}
        for i, partial in enumerate(partial_formulae):
            ind_dict = {}
            ind = partial_indices[i]
            if str(ind) not in partial_derivatives_dict[str(dim)]:
                ind_dict[str(ind)] = partial
                partial_derivatives_dict[str(dim)].update(ind_dict)

        # with open('matrix_data_jsons/partial_derivatives.json', 'w') as f:
        #     json.dump(partial_derivatives_dict, f)

    elif str(dim) in partial_derivatives_dict:
        for i, partial in enumerate(partial_formulae):
            ind_dict = {}
            ind = partial_indices[i]
            if str(ind) not in partial_derivatives_dict[str(dim)]:
                ind_dict[str(ind)] = partial
                partial_derivatives_dict[str(dim)].update(ind_dict)

        # with open('matrix_data_jsons/partial_derivatives.json', 'w') as f:
        #     json.dump(partial_derivatives_dict, f)


    all_hyperplane_eqns = []
    all_partial_derivatives = []

    # iterate through the vertices of each simplex
    for vertices in all_vertices:

        # create edge vectors by subtracting the first coordinate of each vertex from the remaining coordinates
        vector_list = []
        for i in range(dim-1):
            vector = np.transpose(vertices[i+1] - vertices[0])
            vector = vector.round(5)
            vector = np.array(vector)
            vector_list.append(vector)

        # create a matrix where the vectors are stored as columns
        def create_matrix(vectors):
            matrix = np.column_stack(vectors)
            return matrix
        
        matrix = create_matrix(vector_list)

        # function that takes in a 2D matrix reads each element column wise and returns a list of all the elements
        def read_matrix(matrix):
            elements = []
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[0]):
                    elements.append(matrix[j, i])
            return elements
        
        # convert symbolic and numeric matrices to lists
        var_ele = read_matrix(A)
        mat_ele = read_matrix(matrix)

        # initialize an empty dictionary to map the symbolic variables to the numeric values
        var_dict = {}

        for key, value in zip(var_ele, mat_ele):
            var_dict[key] = value

        coord_dict = {}
        for key, value in zip(c, vertices[0]):
            coord_dict[key] = value

        # concatenate var_dict and coord_dict
        var_dict.update(coord_dict)

        # convert all the keys in the var_dict to strings
        var_dict = {str(key): value for key, value in var_dict.items()}
        
        if evaluate_eqns:
            # extract the hyperplane equation from the hyperplane_eqn_dict using the dim as the key
            hyperplane_eqn = hyperplane_eqn_dict[str(dim)]

            safe_globals = {
                "math": math,
                "__builtins__": {},
            }
            
            # add the var_dict to the safe_globals
            safe_globals.update(var_dict)

            # evaluate the hyperplane equation
            eqn = eval(hyperplane_eqn, safe_globals)
            all_hyperplane_eqns.append(eqn)
        
        if evaluate_partials:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # extract the partial derivatives from the partial_derivatives_dict using the dim as the key
                partial_dict = partial_derivatives_dict[str(dim)]
                partials = []
                for i in partial_indices:
                    partial = partial_dict[str(i)]
                    safe_globals = {
                        "math": math,
                        "__builtins__": {},
                    }
                    
                    # add the var_dict to the safe_globals
                    safe_globals.update(var_dict)

                    # evaluate the partial derivative
                    part = eval(partial, safe_globals)
                    part = part*multiplier
                    partials.append(part)
                    all_partial_derivatives.append(partials[0])

    return all_hyperplane_eqns, all_partial_derivatives


def gen_hyperplane_eqns2(points, lower_hull=[], direct_vertices=False, multiplier=1, partial_indices=[0], evaluate_eqns=False, evaluate_partials=True):
    dim = points.shape[1]

    if direct_vertices:
        all_vertices = np.array([points])
    else:
        all_vertices = [points[simplex] for simplex in lower_hull]
        all_vertices = np.array(all_vertices)

    x = np.array([smp.Symbol(f'x{i}') for i in range(dim)]).T

    A = np.array([[smp.Symbol(f'a{j}{i}') for j in range(dim)] for i in range(dim - 1)]).T
    c = np.array([smp.Symbol(f'c{i}') for i in range(dim)]).T

    M = np.hstack((A, x[:, np.newaxis]))
    final_M = smp.Matrix(M)
    sym_normal_vec = smp.det(final_M).simplify()

    normal_vec = np.array([sym_normal_vec.coeff(f'x{i}') for i in range(dim)])
    intercept = -np.dot(normal_vec, c)
    hyperplane_eqn = smp.Add(sym_normal_vec, intercept)
    solution_form = smp.solve(hyperplane_eqn, f'x{dim-1}')

    partial_formulae = [smp.diff(solution_form[0], f'x{i}') for i in range(dim - 1)]
    partial_formulae = [partial_formulae[i] for i in partial_indices]

    x_list = list(x)
    hyperplane_eqn = smp.collect(hyperplane_eqn, x_list)
    remaining_expr = hyperplane_eqn

    coefficients = [hyperplane_eqn.coeff(xi) for xi in x_list]
    for xi in x_list:
        remaining_expr -= hyperplane_eqn.coeff(xi) * xi

    hyperplane_coeffs = coefficients + [remaining_expr]

    hyperplane_eqn_dict = {}
    if os.path.exists('matrix_data_jsons/partial_derivatives.json'):
        with open('matrix_data_jsons/partial_derivatives.json', 'r') as f:
            hyperplane_eqn_dict = json.load(f)

    partial_derivatives_dict = {}
    if os.path.exists('matrix_data_jsons/partial_derivatives.json'):
        with open('matrix_data_jsons/partial_derivatives.json', 'r') as f:
            partial_derivatives_dict = json.load(f)

    all_hyperplane_eqns = []
    all_partial_derivatives = []

    for vertices in all_vertices:
        vector_list = [vertices[i + 1] - vertices[0] for i in range(dim - 1)]
        vector_list = [vector.round(5) for vector in vector_list]

        matrix = np.column_stack(vector_list)

        var_ele = [str(var) for var in A.flatten()]
        mat_ele = matrix.flatten()
        var_dict = {var: val for var, val in zip(var_ele, mat_ele)}

        coord_dict = {str(key): val for key, val in zip(c, vertices[0])}
        var_dict.update(coord_dict)

        if evaluate_eqns:
            eqn = eval(hyperplane_eqn_dict[str(dim)], {"math": math, "__builtins__": {}}, var_dict)
            all_hyperplane_eqns.append(eqn)

        if evaluate_partials:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                partial_dict = partial_derivatives_dict.get(str(dim), {})
                partials = []
                for i in partial_indices:
                    partial = partial_dict.get(str(i), "0")
                    part = eval(partial, {"math": math, "__builtins__": {}}, var_dict)
                    part = part * multiplier
                    partials.append(part)
                if partials:
                    all_partial_derivatives.append(partials[0])

    return all_hyperplane_eqns, all_partial_derivatives


def direct_lowerhull(points):
    hull = ConvexHull(points)
    lowerhull = []
    for eq, simplex in zip(hull.equations, hull.simplices):
        normal = eq[:-1]
        if normal[-1] > 0: 
            lowerhull.append(simplex)
    
    return lowerhull



def gliq_lowerhull(points, liq_points, intermetallics):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    # Input: points = array of coordinates of the points in the Xi-S-H space
    # Output: simplices = array of simplices that form the lower convex hull of the Xi-S-H space
    
    # determine the dimensionality of the points
    dim = points.shape[1]

    sub_points = points[:, :-1]

    sub_hull = ConvexHull(sub_points)

    sub_hull_points = sub_points[sub_hull.vertices]

    center = np.mean(sub_hull_points, axis=0)

    # add center to sub_hull_points
    sub_hull_points = np.vstack((sub_hull_points, center))

    h_index = dim-1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    h_min = np.min(h_data)
    upper_bound = h_max + 4*(abs(h_max - h_min))

    # find column length of sub_hull_points
    sub_hull_points_len = sub_hull_points.shape[0]

    # create an upper_bound column the same length as sub_hull_points
    upper_bound_col = np.full((sub_hull_points_len, 1), upper_bound)

    # add the upper_bound_col to the sub_hull_points
    fake_points = np.hstack((sub_hull_points, upper_bound_col))

    # change the entry in the last row and last column to twice its value
    fake_points[-1, -1] = fake_points[-1, -1] + (0.1*upper_bound)

    new_points = np.vstack((points, fake_points))

    # take the total convex hull
    new_hull = ConvexHull(new_points)

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull_filter1 = []
    lower_hull_filter2 = []

    def check_common_rows(arr1, arr2):
        # print('here')
        return any((arr1 == row).all(axis=1).any() for row in arr2)
    
    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fake_points):
            lower_hull_filter1.append(simplex)
    

    for simplex in lower_hull_filter1:
        vertices = points[simplex]
        count = 0
        for vertex in vertices:
            for intermetallic in intermetallics:
                if (vertex == intermetallic).all():
                    count += 1
        if count < 4:
            lower_hull_filter2.append(simplex)

    arr_lowerhull = np.array(lower_hull_filter2)  # change to lower_hull to exclude misc gaps

    return arr_lowerhull

def gliq_lowerhull3(points, vertical_simplices = False):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    # Input: points = array of coordinates of the points in the Xi-S-H space
    # Output: simplices = array of simplices that form the lower convex hull of the Xi-S-H space
    
    # determine the dimensionality of the points
    dim = points.shape[1]

    sub_points = points[:, :-1]

    sub_hull = ConvexHull(sub_points)

    sub_hull_points = sub_points[sub_hull.vertices]

    center = np.mean(sub_hull_points, axis=0)

    # add center to sub_hull_points
    sub_hull_points = np.vstack((sub_hull_points, center))

    h_index = dim-1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    h_min = np.min(h_data)
    upper_bound = h_max + 10*(abs(h_max - h_min))

    # find column length of sub_hull_points
    sub_hull_points_len = sub_hull_points.shape[0]

    # create an upper_bound column the same length as sub_hull_points
    upper_bound_col = np.full((sub_hull_points_len, 1), upper_bound)

    # add the upper_bound_col to the sub_hull_points
    fake_points = np.hstack((sub_hull_points, upper_bound_col))

    # change the entry in the last row and last column to twice its value
    fake_points[-1, -1] = fake_points[-1, -1] + (0.5*upper_bound)

    new_points = np.vstack((points, fake_points))

    # scale h values of new_points
    new_points[:, -1] = new_points[:, -1]

    # take the total convex hull
    new_hull = ConvexHull(new_points)

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull_filter1 = []

    def check_common_rows(arr1, arr2):
        return any((arr1 == row).all(axis=1).any() for row in arr2)
    
    def has_uniform_column(np_matrix):
        # Check if any column has all the same values
        return np.any(np.all(np_matrix == np_matrix[0, :], axis=0))

    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fake_points):
            coords = new_points[simplex]
            x_coords = coords[:, :-1]
            # check if the simplex has any uniform columns to get rid of "vertical" simplices
            if vertical_simplices == False:
                if not has_uniform_column(x_coords):
                    lower_hull_filter1.append(simplex)
            else:
                lower_hull_filter1.append(simplex)

    arr_lowerhull = np.array(lower_hull_filter1) 

    return arr_lowerhull

def gliq_lowerhull2(points, liq_points, intermetallics):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    dim = points.shape[1]
    sub_points = points[:, :-1]
    sub_hull = ConvexHull(sub_points)
    sub_hull_points = sub_points[sub_hull.vertices]
    center = np.mean(sub_hull_points, axis=0)
    
    sub_hull_points = np.vstack((sub_hull_points, center))
    h_index = dim - 1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    h_min = np.min(h_data)
    upper_bound = h_max + 4 * (abs(h_max - h_min))

    sub_hull_points_len = sub_hull_points.shape[0]
    upper_bound_col = np.full((sub_hull_points_len, 1), upper_bound)
    fake_points = np.hstack((sub_hull_points, upper_bound_col))
    fake_points[-1, -1] = fake_points[-1, -1] + (0.1 * upper_bound)
    
    new_points = np.vstack((points, fake_points))
    new_hull = ConvexHull(new_points)

    lower_hull_filter1 = []
    lower_hull_filter2 = []

    # Convert intermetallics into a set of tuples for faster lookup
    intermetallic_set = set(map(tuple, intermetallics))

    # Revert to row-wise comparison using the original approach, not vectorized
    def check_common_rows(arr1, arr2):
        for row in arr1:
            if any(np.array_equal(row, arr2_row) for arr2_row in arr2):
                return True
        return False

    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fake_points):
            lower_hull_filter1.append(simplex)

    # For filtering based on intermetallics, also revert to a safer precision-aware comparison
    for simplex in lower_hull_filter1:
        vertices = points[simplex]
        count = 0
        for vertex in vertices:
            for intermetallic in intermetallic_set:
                # Use np.allclose to avoid floating-point precision issues
                if np.allclose(vertex, intermetallic, atol=1e-8):
                    count += 1
        if count < 4:
            lower_hull_filter2.append(simplex)

    arr_lowerhull = np.array(lower_hull_filter2)
    return arr_lowerhull



def gen_xE_lowerhull(points, vertical_simplices = False):
    dim = points.shape[1]

    # initialize bounds for Xi
    x_list = []

    for i in range(dim):
        sublist = []
        for j in range(dim-1):
            if i == 0:
                sublist.append(0)
            else:
                if j == i-1:
                    sublist.append(1)
                else:
                    sublist.append(0)
            
        x_list.append(sublist)

    # initialize bounds for E
    e_index = dim-1

    e_data = points[:, e_index]
    e_max = np.max(e_data)
    e_min = np.min(e_data)
    upper_bound = e_max + 2*(abs(e_max - e_min))


    # create a list of fictitious points
    fict_coords = []
    for i in range(dim):
        fict_coord = []
        for j in range(dim-1):
            fict_coord.append(x_list[i][j])
        fict_coord.append(upper_bound)
        fict_coords.append(fict_coord)

    fict_coords = np.array(fict_coords)

    new_points = np.vstack((points, fict_coords))

    # take the total convex hull
    new_hull = ConvexHull(new_points)

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull = []
    def check_common_rows(arr1, arr2):
        return any((arr1 == row).all(axis=1).any() for row in arr2)
    
    def has_uniform_column(np_matrix):
        # Check if any column has all the same values
        return np.any(np.all(np_matrix == np_matrix[0, :], axis=0))

    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fict_coords):
            coords = new_points[simplex]
            x_coords = coords[:, :-1]
            # check if the simplex has any uniform columns to get rid of "vertical" simplices
            if vertical_simplices == False:
                if not has_uniform_column(x_coords):
                    lower_hull.append(simplex)
            else:
                lower_hull.append(simplex)

    arr_lowerhull = np.array(lower_hull)

    return arr_lowerhull


def gen_xME_lowerhull(points, vertical_simplices = False):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    # Input: points = array of coordinates of the points in the Xi-S-H space
    # Output: simplices = array of simplices that form the lower convex hull of the Xi-S-H space

    # determine the dimensionality of the points
    dim = points.shape[1]

    sub_points = points[:, :-1]

    sub_hull = ConvexHull(sub_points)

    # find the points on the sub_hull
    sub_hull_points = sub_points[sub_hull.vertices]

    # find geometric center of sub_points
    center = np.mean(sub_hull_points, axis=0)

    # add center to sub_hull_points
    sub_hull_points = np.vstack((sub_hull_points, center))
    h_index = dim-1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    h_min = np.min(h_data)
    upper_bound = h_max + 2*(abs(h_max - h_min))
    
    # find column length of sub_hull_points
    sub_hull_points_len = sub_hull_points.shape[0]

    # create an upper_bound column the same length as sub_hull_points
    upper_bound_col = np.full((sub_hull_points_len, 1), upper_bound)

    # add the upper_bound_col to the sub_hull_points
    fake_points = np.hstack((sub_hull_points, upper_bound_col))

    # change the entry in the last row and last column to twice its value
    fake_points[-1, -1] = fake_points[-1, -1] + (0.1*upper_bound)

    new_points = np.vstack((points, fake_points))

    # take the total convex hull
    new_hull = ConvexHull(new_points)

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull = []
    def check_common_rows(arr1, arr2):
        return any((arr1 == row).all(axis=1).any() for row in arr2)
    
    def has_uniform_column(np_matrix):
        # Check if any column has all the same values
        return np.any(np.all(np_matrix == np_matrix[0, :], axis=0))

    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fake_points):
            coords = new_points[simplex]
            x_coords = coords[:, :-2]
            # check if the simplex has any uniform columns to get rid of "vertical" simplices
            if vertical_simplices == False:
                if not has_uniform_column(x_coords):
                    lower_hull.append(simplex)
            else:
                lower_hull.append(simplex)

    arr_lowerhull = np.array(lower_hull)

    return arr_lowerhull, new_points


def hyperplane_eqns(points, lower_hull, multiplier, partial_indices):
    # Function to calculate hyperplane equations and partial derivatives of lower hull simplices
    # Input: points = array of coordinates of the points in the Xi-S-H space
    #        lower_hull = array of simplices that form the lower convex hull of the Xi-S-H space
    #        multiplier = multiplier to correct scaled enthalpy values
    # output: all_partial_derivatives = list of all the partial derivatives of the hyperplane equations of the simplices

    # determine the dimensionality of the points
    dim = points.shape[1]
    
    # extract the vertices of each simplex of the lower hull
    all_vertices = []
    for simplex in lower_hull:
        vertices = points[simplex]
        all_vertices.append(vertices)
    all_vertices = np.array(all_vertices)
    
    # initialize array of symbolic variables that form the basis of the coordinate space
    x = []
    for i in range(dim):
        x.append(smp.Symbol('x{}'.format(i)))
    x = np.array(x)
    x = np.transpose(x)

    # initialize array of symbolic variables that represent the coordinates of the vertices
    A = []
    for i in range(dim-1):
        a = []
        for j in range(dim):
            a.append(smp.Symbol('a{}{}'.format(j, i)))
        A.append(a)
    A = np.array(A)
    A = np.transpose(A)

    # intialize a symbolic vector that points to one of the vertices of the simplex
    c = []
    for i in range(dim):
        c.append(smp.Symbol('c{}'.format(i)))
    c = np.array(c)
    c = np.transpose(c)
  
    # horizontally stack the A and x matrices
    M = np.hstack((A, x[:, np.newaxis]))

    # convert the matrix to a sympy matrix
    final_M = smp.Matrix(M)
    
    # analytically solve for the general symbolic equation of the hyperplane 
    sym_normal_vec = smp.det(final_M).simplify()

    # extract the coefficients of the normal vector and put them in an array
    normal_vec = []
    for i in range(dim):
        normal_vec.append(sym_normal_vec.coeff('x{}'.format(i)))
    normal_vec = np.array(normal_vec)

    # symbolically calculate the dot product of the normal vector and the coordinate vector
    intercept = -np.dot(normal_vec, c)
    
    hyperplane_eqn = smp.Add(sym_normal_vec, intercept)

    # express the equation in terms of x_dim
    solution_form = smp.solve(hyperplane_eqn, 'x{}'.format(dim-1))

    # analytically solve for the partial derivatives of the hyperplane equation
    partial_formulae = []

    for i in range(dim-1):
        partial_formulae.append(smp.diff(solution_form[0], 'x{}'.format(i)))

    # create a new list of partial formulae that only contains the elements from the list of partial_indices
    partial_formulae = [partial_formulae[i] for i in partial_indices]
    print("partial derivatives: ", partial_formulae)

    # initialize an empty list to store all the to be computed partial derivatives
    all_partial_derivatives = []

    # iterate through the vertices of each simplex
    for vertices in all_vertices:

        # create edge vectors by subtracting the first coordinate of each vertex from the remaining coordinates
        vector_list = []
        for i in range(dim-1):
            vector = np.transpose(vertices[i+1] - vertices[0])
            vector = vector.round(5)
            vector = smp.Matrix(vector)
            vector_list.append(vector)


        # create a matrix where the vectors are stored as columns
        def create_matrix(vectors):
            matrix = np.column_stack(vectors)
            return matrix
        
        matrix = create_matrix(vector_list)
        
        # function that takes in a 2D matrix reads each element column wise and returns a list of all the elements
        def read_matrix(matrix):
            elements = []
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[0]):
                    elements.append(matrix[j, i])
            return elements
        
        # convert symbolic and numeric matrices to lists
        var_ele = read_matrix(A)
        mat_ele = read_matrix(matrix)

        # initialize an empty dictionary to map the symbolic variables to the numeric values
        var_dict = {}
        for key, value in zip(var_ele, mat_ele):
            var_dict[key] = value

        coord_dict = {}
        for key, value in zip(c, vertices[0]):
            coord_dict[key] = value

        # concatenate var_dict and coord_dict
        var_dict.update(coord_dict)

        
        # substitute the symbolic variables with the numeric values to evaluate the partial derivatives
        partials = []
        for form in partial_formulae:
            # if TypeError is raised, move on to the next iteration while appending a 'nan' to the list of partials
            try:
                eqn = form.subs(var_dict) 
                val = smp.N(eqn)
                val = float(val)
                val = val*multiplier
                # take real part of the value
                partials.append(val)
            except TypeError:
                partials.append('nan')
                continue 
        
        # append  partial derivatives to the list of all  partial derivatives
        all_partial_derivatives.append(partials)