import { Tensor1D } from '@tensorflow/tfjs'
import { Scikit1D, Scikit2D } from '../types'
import { Splitter } from './splitter'
// from https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/tree/_tree.pxd
export class Node {
  left_child: number = -1 // id of left child
  right_child: number = -1 // id of the right child of the node
  feature: number = -1 // Feature used for splitting the node
  threshold: number = -1 //Threshold value at the node
  impurity: number = -1 // Impurity of the node (i.e., the value of the criterion)
  n_node_samples: number = -1 // Number of samples at the node
  weighted_n_node_samples: number = -1 //Weighted number of samples at the node
}
export class Tree {
  n_features: number = -1
  n_classes: number[] = [];
  n_outputs: number = -1
  max_n_classes: number = -1
  max_depth: number = -1
  node_count: number = -1
  capacity: number = -1
  nodes: Node[] = []
  value: number = -1
  value_stride: number = -1
  // Methods
  // cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
  //                       SIZE_t feature, double threshold, double impurity,
  //                       SIZE_t n_node_samples,
  //                       double weighted_n_node_samples) nogil except -1
  // cdef int _resize(self, SIZE_t capacity) nogil except -1
  // cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

  // cdef np.ndarray _get_value_ndarray(self)
  // cdef np.ndarray _get_node_ndarray(self)

  // cpdef np.ndarray predict(self, object X)

  // cpdef np.ndarray apply(self, object X)
  // cdef np.ndarray _apply_dense(self, object X)
  // cdef np.ndarray _apply_sparse_csr(self, object X)

  // cpdef object decision_path(self, object X)
  // cdef object _decision_path_dense(self, object X)
  // cdef object _decision_path_sparse_csr(self, object X)

  // cpdef compute_feature_importances(self, normalize=*)
  constructor(n_features: number, n_classes: Tensor1D,
    n_outputs: number){
      // """Constructor."""
      // Input/Output layout
      this.n_features = n_features
      this.n_outputs = n_outputs

      // TODO?
      this.max_n_classes = n_classes.max().dataSync()[0];
      this.value_stride = n_outputs * this.max_n_classes
      this.n_classes = new Array(n_outputs);
    for (let k = 0; k < n_outputs; k++) {
      this.n_classes[k] = n_classes.dataSync()[k];
    }
      // Inner structures
      this.max_depth = 0
      this.node_count = 0
      this.capacity = 0
      this.value = -1; // should be null?
      this.nodes = [];// shuld be null?
    }
    get n_leaves() {
      // TODO
      return -1;
    }
}
export abstract class TreeBuilder {
  splitter!: Splitter // Splitting algorithm
  min_samples_split: number = -1 // Minimum number of samples in an internal node
  min_samples_leaf: number = -1 // Minimum number of samples in a leaf
  min_weight_leaf: number = -1 // Minimum weight in a leaf
  max_depth: number = -1 // Maximal tree depth
  min_impurity_decrease: number = -1 // Impurity threshold for early stopping
  // build(self, Tree tree, object X, np.ndarray y,
  //   np.ndarray sample_weight=None)
  abstract build(
    tree: Tree,
    X: any,
    y: Scikit1D | Scikit2D,
    sample_weight?: Scikit1D
  ): void
}
