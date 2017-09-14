The **word_graphs API** offers an interface for generating and storing word graphs via the WordGraph class, extracting, storing, and classifying subgraphs of word graphs, and analyzing the properties of word graphs and their subgraphs. Using Numba, the API also includes several CUDA interfaces for fast GPU calculation of word graphs and their subgraphs. 

Currently, only the following subgraphs are supported: 3-paths, 4-paths, triangles, squares, and cubes.

**word_graphs** - Defines a class WordGraph used for generating word graphs from a given set of words.

**word_graphs_gpu** - A CUDA Python implementation of 'word_graphs' using Numba for speeding up the word graph generation process.

**subgraphs** - Contains functions for finding all subgraphs of a given type within a word graph.

**subgraphs_gpu** - A CUDA Python implementation of 'subgraphs' using Numba for speeding up the subgraph finding process.

**vertex_induced_subgraphs** - Contains functions for extracting and storing several types of induced subgraphs from a given word graph.

**graph_analysis** - Contains functions for analyzing a given graph and locating certain types of paths using NetworkX.

**subgraph_analysis** - Contains functions for analyzing and classifying subgraphs of a word graph based on its directed edges---currently only supports 'square' subgraphs.

**io** - Input/output functions for the word_graphs API, including tools for storing and retrieving word graphs and their subgraphs, plus subgraph statistics. 