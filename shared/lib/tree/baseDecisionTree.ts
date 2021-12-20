import { tf } from "globals";
import { Scikit1D, Scikit2D } from "../types";
import { convertToNumericTensor1D_2D, convertToNumericTensor2D, convertToTensor } from "../utils";
import { max, reshape } from "mathjs";
import { Tree } from "./tree";

export interface BaseDecisionTreeParams {
  criterion: string;
  splitter: string;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf: number;
  min_weight_fraction_leaf: number;
  max_features: number;
  max_leaf_nodes: number;
  random_state: number;
  min_impurity_decrease: number;
  // TODO: should be dict, list of dict or “balanced”, default=None
  class_weight: null | string | Object,
  ccp_alpha: number;
  
}
export abstract class BaseDecisionTree {//(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
  // 
  n_features_in_: number=-1;
  n_outputs_: number=-1;
  classes_: tf.Tensor2D[]=[];
  n_classes_: number[]=[];
  criterion: string;
  ccp_alpha: number;
  class_weight: null|string | Object;
  min_impurity_decrease: number;
  random_state: number;
  max_leaf_nodes: number;
  max_features: number;
  min_weight_fraction_leaf: number;
  min_samples_leaf: number;
  min_samples_split: number;
  max_depth: number;
  splitter: string;

  tree_!: Tree

  constructor(
    {criterion,
    splitter,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf,
    max_features,
    max_leaf_nodes,
    random_state,
    min_impurity_decrease,
    class_weight=null,
    ccp_alpha}:BaseDecisionTreeParams
  ) {
    this.criterion = criterion;
    this.splitter = splitter;
    this.max_depth = max_depth;
    this.min_samples_split = min_samples_split;
    this.min_samples_leaf = min_samples_leaf;
    this.min_weight_fraction_leaf = min_weight_fraction_leaf;
    this.max_features = max_features;
    this.max_leaf_nodes = max_leaf_nodes;
    this.random_state = random_state;
    this.min_impurity_decrease = min_impurity_decrease;
    this.class_weight = class_weight;
    this.ccp_alpha = ccp_alpha;

  }

  /**
   * Return the depth of the decision tree.
   The depth of a tree is the maximum distance between the root
      and any leaf.
      Returns
      -------
      this.tree_.max_depth : number
          The maximum depth of the tree.
   * @returns 
   */
  getDepth() {
    // TODO check_is_fitted(this)
    return this.tree_.max_depth;
  }
  /**
   * Return the number of leaves of the decision tree.
      Returns
      -------
      this.tree_.n_leaves : int
          Number of leaves.
   */
  getNLeaves() {

      // check_is_fitted(self)
      return this.tree_.n_leaves
  }
  /**
   * 
   * @param X The training input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csc_matrix.
   * @param y The target values (class labels) as integers or strings.
   * @param sample_weight Sample weights. If None, then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node. Splits are also ignored if they would result in any single class carrying a negative weight in either child node.
   * @param check_input Allow to bypass several input checking. Don’t use this parameter unless you know what you do.
   */
  fit(
    X: Scikit2D, y: Scikit1D | Scikit2D, sample_weight: Scikit1D, check_input: boolean = true
  ) {
      // random_state = check_random_state(self.random_state)

      if (this.ccp_alpha < 0.0) throw Error("ccp_alpha must be greater than or equal to 0")

      // TODO
      // if (check_input){
      //     // # Need to validate separately here.
      //     // # We can't pass multi_ouput=True because that would allow y to be
      //     // # csr.
      //     check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
      //     check_y_params = dict(ensure_2d=False, dtype=None)
      //     X, y = self._validate_data(
      //         X, y, validate_separately=(check_X_params, check_y_params)
      //     )
      //     if issparse(X):
      //         X.sort_indices()

      //         if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
      //             raise ValueError(
      //                 "No support for np.int64 index based sparse matrices"
      //             )

      //     if self.criterion == "poisson":
      //         if np.any(y < 0):
      //             raise ValueError(
      //                 "Some value(s) of y are negative which is"
      //                 " not allowed for Poisson regression."
      //             )
      //         if np.sum(y) <= 0:
      //             raise ValueError(
      //                 "Sum of y is not positive which is "
      //                 "necessary for Poisson regression."
      //             )
      // }
      // Determine output settings
      let XTwoD = convertToNumericTensor2D(X)
      let yOneD = convertToNumericTensor1D_2D(y)
      // TODO: n_features_in_ is something every model has i think?
      let n_samples = XTwoD.shape[0]
      this.n_features_in_ = XTwoD.shape[1]
      // is_classification = is_classifier(self)
      // TODO:
      let is_classification = true;

      // y = np.atleast_1d(y)
      // expanded_class_weight = null
      let yTwoD: tf.Tensor2D;
      if (yOneD.shape.length == 1) {
          // reshape is necessary to preserve the data contiguity against vs
          // [:, np.newaxis] that does not.
          yTwoD = yOneD.reshape([-1, 1]);
          // y = np.reshape(y, (-1, 1))
      } else {
        yTwoD = yOneD as tf.Tensor2D;
      }

      // tODO: ?
      this.n_outputs_ = yTwoD.shape[1]

      if (is_classification){
          //TODO check_classification_targets(y)
          // y = np.copy(y)
          yTwoD = yTwoD.clone()

          this.classes_ = []
          this.n_classes_ = []
          let y_original: typeof yTwoD;
          if (this.class_weight != null) {
            y_original = yTwoD.clone();
          }
          let y_encoded = tf.zeros(yTwoD.shape, "int32") // y_encoded = np.zeros(y.shape, dtype=int)
          for (let k = 0; k < this.n_outputs_; k++) {
            let unique = tf.unique(yTwoD.slice([0, k], [yTwoD.shape[0], 1]))  // np.unique(y[:, k], return_inverse=True)
            let classes_k = unique.values;
            //TODO check!  y_encoded[:, k] = unique.indices
            y_encoded.slice([0, k], [yTwoD.shape[0], 1]) = unique.indices
            this.classes_.push(classes_k)
            this.n_classes_.push(classes_k.shape[0])
          }
          yTwoD = y_encoded;

          if (this.class_weight != null) {
            // expanded_class_weight = compute_sample_weight(
            //   self.class_weight, y_original
            // )
          }

          this.n_classes_ = convertToTensor(this.n_classes_, undefined, "int32"); // np.array(self.n_classes_, dtype=np.intp)
      }
      // if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
      //     y = np.ascontiguousarray(y, dtype=DOUBLE)

      // Check parameters
      // max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
      let max_depth = Infinity // TODO 2^32?
      if (this.max_depth != null) this.max_depth;
      // max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes
      let max_leaf_nodes = -1;
      if (this.max_leaf_nodes != null) max_leaf_nodes = this.max_leaf_nodes;

      // if isinstance(self.min_samples_leaf, numbers.Integral):
      //     if not 1 <= self.min_samples_leaf:
      //         raise ValueError(
      //             "min_samples_leaf must be at least 1 or in (0, 0.5], got %s"
      //             % self.min_samples_leaf
      //         )
      //     min_samples_leaf = self.min_samples_leaf
      // else:  # float
      //     if not 0.0 < self.min_samples_leaf <= 0.5:
      //         raise ValueError(
      //             "min_samples_leaf must be at least 1 or in (0, 0.5], got %s"
      //             % self.min_samples_leaf
      //         )
      //     min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))
      const min_samples_leaf = this.min_samples_leaf
      // TODO: some number formatting for min_samples_split
      // if isinstance(self.min_samples_split, numbers.Integral):
      //     if not 2 <= self.min_samples_split:
      //         raise ValueError(
      //             "min_samples_split must be an integer "
      //             "greater than 1 or a float in (0.0, 1.0]; "
      //             "got the integer %s"
      //             % self.min_samples_split
      //         )
      //     min_samples_split = self.min_samples_split
      // else:  # float
      //     if not 0.0 < self.min_samples_split <= 1.0:
      //         raise ValueError(
      //             "min_samples_split must be an integer "
      //             "greater than 1 or a float in (0.0, 1.0]; "
      //             "got the float %s"
      //             % self.min_samples_split
      //         )
      //     min_samples_split = int(ceil(self.min_samples_split * n_samples))
      //     min_samples_split = max(2, min_samples_split)

      // min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
      const min_samples_split = max(this.min_samples_split, 2 * min_samples_leaf);

      // TODO: some number formatting for max_features
      // if isinstance(self.max_features, str):
      //     if self.max_features == "auto":
      //         if is_classification:
      //             max_features = max(1, int(np.sqrt(self.n_features_in_)))
      //         else:
      //             max_features = self.n_features_in_
      //     elif self.max_features == "sqrt":
      //         max_features = max(1, int(np.sqrt(self.n_features_in_)))
      //     elif self.max_features == "log2":
      //         max_features = max(1, int(np.log2(self.n_features_in_)))
      //     else:
      //         raise ValueError(
      //             "Invalid value for max_features. "
      //             "Allowed string values are 'auto', "
      //             "'sqrt' or 'log2'."
      //         )
      // elif self.max_features is None:
      //     max_features = self.n_features_in_
      // elif isinstance(self.max_features, numbers.Integral):
      //     max_features = self.max_features
      // else:  # float
      //     if self.max_features > 0.0:
      //         max_features = max(1, int(self.max_features * self.n_features_in_))
      //     else:
      //         max_features = 0
      let max_features = this.max_features

      if (yTwoD.shape[0] != n_samples){
          throw Error(`Number of labels=${yOneD.shape[0]} does not match number of samples=${n_samples}`);
      }
      // TODO: value checking
      // if not 0 <= self.min_weight_fraction_leaf <= 0.5:
      //     raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
      // if max_depth <= 0:
      //     raise ValueError("max_depth must be greater than zero. ")
      // if not (0 < max_features <= self.n_features_in_):
      //     raise ValueError("max_features must be in (0, n_features]")
      // if not isinstance(max_leaf_nodes, numbers.Integral):
      //     raise ValueError(
      //         "max_leaf_nodes must be integral number but was %r" % max_leaf_nodes
      //     )
      // if -1 < max_leaf_nodes < 2:
      //     raise ValueError(
      //         ("max_leaf_nodes {0} must be either None or larger than 1").format(
      //             max_leaf_nodes
      //         )
      //     )

      // if sample_weight is not None:
      //     sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

      // if expanded_class_weight is not None:
      //     if sample_weight is not None:
      //         sample_weight = sample_weight * expanded_class_weight
      //     else:
      //         sample_weight = expanded_class_weight

      // Set min_weight_leaf from min_weight_fraction_leaf
      // if sample_weight is None:
      //     min_weight_leaf = self.min_weight_fraction_leaf * n_samples
      // else:
      //     min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

      // if self.min_impurity_decrease < 0.0:
      //     raise ValueError("min_impurity_decrease must be greater than or equal to 0")

      // # Build tree
      const criterion_name = this.criterion
      let criterion: any;
      // if not isinstance(criterion, Criterion): // overloading thing here, we can ignore for now
      if (is_classification) {
        criterion = CRITERIA_CLF[this.criterion](
                this.n_outputs_, this.n_classes_
            )
      }
      else {
        criterion = CRITERIA_REG[this.criterion](this.n_outputs_, n_samples)
      }

      // SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
      const SPLITTERS = DENSE_SPLITTERS

      const splitter_name = this.splitter
      if not isinstance(self.splitter, Splitter):
          splitter = SPLITTERS[self.splitter](
              criterion,
              self.max_features_,
              min_samples_leaf,
              min_weight_leaf,
              random_state,
          )

      if is_classifier(self):
          self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
      else:
          self.tree_ = Tree(
              self.n_features_in_,
              # TODO: tree shouldn't need this in this case
              np.array([1] * self.n_outputs_, dtype=np.intp),
              self.n_outputs_,
          )

      # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
      if max_leaf_nodes < 0:
          builder = DepthFirstTreeBuilder(
              splitter,
              min_samples_split,
              min_samples_leaf,
              min_weight_leaf,
              max_depth,
              self.min_impurity_decrease,
          )
      else:
          builder = BestFirstTreeBuilder(
              splitter,
              min_samples_split,
              min_samples_leaf,
              min_weight_leaf,
              max_depth,
              max_leaf_nodes,
              self.min_impurity_decrease,
          )

      builder.build(self.tree_, X, y, sample_weight)

      if self.n_outputs_ == 1 and is_classifier(self):
          self.n_classes_ = self.n_classes_[0]
          self.classes_ = self.classes_[0]

      self._prune_tree()

      return self

  def _validate_X_predict(self, X, check_input):
      """Validate the training data on predict (probabilities)."""
      if check_input:
          X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
          if issparse(X) and (
              X.indices.dtype != np.intc or X.indptr.dtype != np.intc
          ):
              raise ValueError("No support for np.int64 index based sparse matrices")
      else:
          # The number of features is checked regardless of `check_input`
          self._check_n_features(X, reset=False)
      return X

  def predict(self, X, check_input=True):
      """Predict class or regression value for X.
      For a classification model, the predicted class for each sample in X is
      returned. For a regression model, the predicted value based on X is
      returned.
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          The input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csr_matrix``.
      check_input : bool, default=True
          Allow to bypass several input checking.
          Don't use this parameter unless you know what you do.
      Returns
      -------
      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The predicted classes, or the predict values.
      """
      check_is_fitted(self)
      X = self._validate_X_predict(X, check_input)
      proba = self.tree_.predict(X)
      n_samples = X.shape[0]

      # Classification
      if is_classifier(self):
          if self.n_outputs_ == 1:
              return self.classes_.take(np.argmax(proba, axis=1), axis=0)

          else:
              class_type = self.classes_[0].dtype
              predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
              for k in range(self.n_outputs_):
                  predictions[:, k] = self.classes_[k].take(
                      np.argmax(proba[:, k], axis=1), axis=0
                  )

              return predictions

      # Regression
      else:
          if self.n_outputs_ == 1:
              return proba[:, 0]

          else:
              return proba[:, :, 0]

  def apply(self, X, check_input=True):
      """Return the index of the leaf that each sample is predicted as.
      .. versionadded:: 0.17
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          The input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csr_matrix``.
      check_input : bool, default=True
          Allow to bypass several input checking.
          Don't use this parameter unless you know what you do.
      Returns
      -------
      X_leaves : array-like of shape (n_samples,)
          For each datapoint x in X, return the index of the leaf x
          ends up in. Leaves are numbered within
          ``[0; self.tree_.node_count)``, possibly with gaps in the
          numbering.
      """
      check_is_fitted(self)
      X = self._validate_X_predict(X, check_input)
      return self.tree_.apply(X)

  def decision_path(self, X, check_input=True):
      """Return the decision path in the tree.
      .. versionadded:: 0.18
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          The input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csr_matrix``.
      check_input : bool, default=True
          Allow to bypass several input checking.
          Don't use this parameter unless you know what you do.
      Returns
      -------
      indicator : sparse matrix of shape (n_samples, n_nodes)
          Return a node indicator CSR matrix where non zero elements
          indicates that the samples goes through the nodes.
      """
      X = self._validate_X_predict(X, check_input)
      return self.tree_.decision_path(X)

  def _prune_tree(self):
      """Prune tree using Minimal Cost-Complexity Pruning."""
      check_is_fitted(self)

      if self.ccp_alpha < 0.0:
          raise ValueError("ccp_alpha must be greater than or equal to 0")

      if self.ccp_alpha == 0.0:
          return

      # build pruned tree
      if is_classifier(self):
          n_classes = np.atleast_1d(self.n_classes_)
          pruned_tree = Tree(self.n_features_in_, n_classes, self.n_outputs_)
      else:
          pruned_tree = Tree(
              self.n_features_in_,
              # TODO: the tree shouldn't need this param
              np.array([1] * self.n_outputs_, dtype=np.intp),
              self.n_outputs_,
          )
      _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

      self.tree_ = pruned_tree

  def cost_complexity_pruning_path(self, X, y, sample_weight=None):
      """Compute the pruning path during Minimal Cost-Complexity Pruning.
      See :ref:`minimal_cost_complexity_pruning` for details on the pruning
      process.
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          The training input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csc_matrix``.
      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The target values (class labels) as integers or strings.
      sample_weight : array-like of shape (n_samples,), default=None
          Sample weights. If None, then samples are equally weighted. Splits
          that would create child nodes with net zero or negative weight are
          ignored while searching for a split in each node. Splits are also
          ignored if they would result in any single class carrying a
          negative weight in either child node.
      Returns
      -------
      ccp_path : :class:`~sklearn.utils.Bunch`
          Dictionary-like object, with the following attributes.
          ccp_alphas : ndarray
              Effective alphas of subtree during pruning.
          impurities : ndarray
              Sum of the impurities of the subtree leaves for the
              corresponding alpha value in ``ccp_alphas``.
      """
      est = clone(self).set_params(ccp_alpha=0.0)
      est.fit(X, y, sample_weight=sample_weight)
      return Bunch(**ccp_pruning_path(est.tree_))

  @property
  def feature_importances_(self):
      """Return the feature importances.
      The importance of a feature is computed as the (normalized) total
      reduction of the criterion brought by that feature.
      It is also known as the Gini importance.
      Warning: impurity-based feature importances can be misleading for
      high cardinality features (many unique values). See
      :func:`sklearn.inspection.permutation_importance` as an alternative.
      Returns
      -------
      feature_importances_ : ndarray of shape (n_features,)
          Normalized total reduction of criteria by feature
          (Gini importance).
      """
      check_is_fitted(self)

      return self.tree_.compute_feature_importances()
}