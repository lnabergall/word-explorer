# -*- coding: utf-8 -*-

import numpy as np
import igraph as ig

import tigraphs2 as tig


class Simplex(tig.BasicNode, object):
    def __init__(self, vertices, **kwargs):
        super(Simplex, self).__init__(**kwargs)
        
        self.vertices = vertices
        self.faces = []
        self.cofaces = []
        self.dimension = len(self.vertices) - 1
        # These are used for initialising the complex:
        self._cfaces = []
        self._children = {}
        self.__parent = None

        
class SimplicialComplex(tig.return_tree_class(), object):
    def __init__(self, maximal_simplices, Vertex=Simplex, update_adj=True, **kwargs):
        super(SimplicialComplex, self).__init__(Vertex=Vertex, **kwargs)
        self.maximal_simplices = maximal_simplices
        self.dimension = max(map(len, self.maximal_simplices)) - 1
        self.n_simplex_dict = {} # Keeps track of simplexes by dimension.
        for i in range(-1, self.dimension+1):
            self.n_simplex_dict[i] = []
        # Create the empty set 'root' of the structure.
        self.create_simplex([])
        self.set_root(self.vertices[0])
        self.vertices[0].label = str([])
        # Initialize complex from maximal simplices
        for ms in self.maximal_simplices:
            self.add_maximal_simplex(ms)
        if update_adj:
            self.update_adjacency()
        
    def create_simplex(self, vertices):
        self.create_vertex(vertices)
        vertex = self.vertices[-1]    
        self.n_simplex_dict[vertex.dimension].append(self.vertices[-1])
        # It is convienient for the simplex to know its index for later on
        # when we create 'vectors' of simplices.
        index = len(self.n_simplex_dict[vertex.dimension]) - 1
        vertex.index = index
        vertex.label = str(vertices)

    def create_child(self, simplex, vertex):
        if vertex in simplex._cfaces:
            return
        simplex._cfaces.append(vertex)
        child_vertices = [v for v in simplex.vertices]
        child_vertices.append(vertex)
        self.create_simplex(child_vertices)
        child = self.vertices[-1]
        child._parent = simplex
        simplex._children[vertex] = child
        self.leaves.add(child)
        self.create_edge(ends=[simplex, child])
        simplex.cofaces.append(child)
        child.faces.append(simplex)

    def add_maximal_simplex(self, vertices, simplex=None, first=True):
        if first:
            simplex = self.get_root()
        
        if len(vertices) >= 1:
            for index in range(len(vertices)):
                vertex = vertices[index]
                self.create_child(simplex, vertex)
                self.add_maximal_simplex(simplex=simplex._children[vertex],
                                         vertices=vertices[index+1:],
                                         first=False)

    def get_simplex(self, address, first=True, simplex=None):
        if first:
            simplex = self.get_root()
        if len(address) == 0:
            return simplex
        if address[0] not in simplex._cfaces:
            return None
        return self.get_simplex(address=address[1:], first=False,
                                simplex=simplex._children[address[0]])

    def _boundary(self, simplex):
        vertices = simplex.vertices
        n = len(vertices)
        boundary = []
        for index in range(n):
            boundary.append(vertices[:index] + vertices[index+1:])
        return boundary

    def add_face_coface(self, face, coface):
        if coface._parent != face:
            self.create_edge([face, coface])
            face.cofaces.append(coface)
            coface.faces.append(face)

    def update_adjacency_simplex(self, simplex):
        boundary_faces = self._boundary(simplex)
        boundary_faces = map(self.get_simplex, boundary_faces)
        for face in boundary_faces:
            self.add_face_coface(face, simplex)

    def update_adjacency(self):
        for i in range(self.dimension+1):
            for simplex in self.n_simplex_dict[i]:
                self.update_adjacency_simplex(simplex)

    def chain_rank(self, p):
        return len(self.n_simplex_dict[p])

    def boundary_column(self, simplex):
        column_size = self.chain_rank(simplex.dimension-1)
        faces = simplex.faces
        column = [0 for i in range(column_size)]
        for face in faces:           
            column[face.index] = 1
        return column
   
    def get_boundary_matrix(self, p, skeleton=None):
        if skeleton == None:
            skeleton = self.dimension
        if ((p > skeleton) or (p < 0)):
            return np.asarray([])
        mat=[self.boundary_column(simplex) for simplex in self.n_simplex_dict[p]]       
        return np.asarray(mat, dtype=int).transpose()

    def plot(self, margin=60, direction='left', label_color='black'):
        A = self.get_adjacency_matrix_as_list()
        g = ig.Graph.Adjacency(A, mode='undirected')
        for vertex in self.vertices:
            if vertex.label !=None:
                # print vertex.label
                index = self.vertices.index(vertex)
                g.vs[index]['label'] = vertex.label
        layout = g.layout("rt", root=0)
        visual_style = {}
        visual_style["vertex_size"] = 20
        visual_style["vertex_label_color"] = label_color
        if direction == 'left':
            visual_style["vertex_label_angle"] = 3
        elif direction == 'right':
            visual_style["vertex_label_angle"] = 0
        elif direction == 'up':
            visual_style["vertex_label_angle"] = 1.5
        elif direction == 'down':
            visual_style["vertex_label_angle"] = 4.5
            
        visual_style["vertex_label_dist"] = 1
        layout.rotate(180)
        ig.plot(g, layout=layout, margin=margin, **visual_style)       

    def reduce_matrix(self, matrix):        
        if np.size(matrix) == 0:
            return [matrix, 0, 0]
        m = matrix.shape[0]
        n = matrix.shape[1]

        def _reduce(x):
            # We recurse through the digonal entries.
            # We move a 1 to the diagonal entry, then
            # knock out any other 1s in the same  col/row.
            # The rank is the number of nonzero pivots,
            # so when we run out of nonzero diagonal entries, we will
            # know the rank.
            nonzero = False
            for i in range(x,m):
                for j in range(x,n):
                    if matrix[i,j] == 1:
                        matrix[[x,i],:] = matrix[[i,x],:]
                        matrix[:,[x,j]] = matrix[:,[j,x]]
                        nonzero = True
                        break
                if nonzero:
                    break
            if nonzero:
                for i in range(x+1, m):
                    if matrix[i,x] == 1:
                        matrix[i,:] = np.logical_xor(matrix[x,:], matrix[i,:])
                for i in range(x+1, n):
                    if matrix[x,i] == 1:
                        matrix[:,i] = np.logical_xor(matrix[:,x], matrix[:,i])
                return _reduce(x+1)
            else:
                return x
        rank = _reduce(0)
        return [matrix, rank, n-rank]    

    def boundary_rank(self, p, skeleton=None):
        if skeleton == None:
            skeleton = self.dimension
        if (p >= skeleton) or (p < 0):
            return 0
        else:
            return self.reduce_matrix(self.get_boundary_matrix(p+1, skeleton))[1]

    def cycle_rank(self, p, skeleton=None):
        if skeleton == None:
            skeleton = self.dimension
        if p == 0:
            return self.chain_rank(0)
        elif (p > skeleton) or (p < 0):
            return 0
        else:
            return self.reduce_matrix(self.get_boundary_matrix(p, skeleton))[2]

    def betti_number(self, p, skeleton=None):
        if skeleton == None:
            skeleton = self.dimension
        return self.cycle_rank(p,skeleton) - self.boundary_rank(p, skeleton)


def jeremy_example(skeleton=2):
    maximal_simplices = [[0,1,2,3], [0,4], [2,4]]
    S = SimplicialComplex(maximal_simplices)
    #S.plot(direction='up', label_color='blue')
    for i in range(4):
        print("Boundary matrix of dimension", i)
        print(S.get_boundary_matrix(i, skeleton))
        print("Reduced boundary matrix of dimension", i)
        print(S.reduce_matrix(S.get_boundary_matrix(i, skeleton))[0])
        print("Betti number", i, "=", S.betti_number(i, skeleton))

#jeremy_example()
