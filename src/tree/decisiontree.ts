import { ImpurityMeasure } from './criterion'
import { Splitter } from './splitter'
import { int } from '../randUtils'
import { r2Score, accuracyScore } from '../metrics/metrics'
import { Split, makeDefaultSplit } from './splitter'
import { assert } from '../typesUtils'
interface NodeRecord {
  start: int
  end: int
  n_samples: int
  depth: int
  parent_id: int
  is_left: boolean
  impurity: number
}

interface Node {
  parent_id: int
  left_child_id: int
  right_child_id: int
  is_left: boolean
  is_leaf: boolean
  impurity: number
  split_feature: int
  threshold: number
  n_samples: int
  value: int[]
}

function SetMaxFeatures(
  max_features: int,
  max_features_method: string,
  feature_data: number[][]
) {
  let n_features = feature_data[0].length
  if (max_features < 1) {
    switch (max_features_method) {
      case 'log2_method':
        max_features = Math.floor(Math.log2(n_features))
        break
      case 'sqrt_method':
        max_features = Math.floor(Math.sqrt(n_features))
        break
      case 'all_method':
        max_features = n_features
        break
    }
  } else if (max_features > n_features) {
    max_features = n_features
  }
  return max_features
}

function argMax(array: number[]) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]
}

class DecisionTree {
  nodes: Node[] = []
  is_built = false
  GetLeafNodes(feature_data: number[][]): int[] {
    let leaf_node_ids: int[] = []
    for (let i = 0; i < feature_data.length; i++) {
      let node_id = 0
      while (!this.nodes[node_id].is_leaf) {
        if (
          feature_data[i][this.nodes[node_id].split_feature] <=
          this.nodes[node_id].threshold
        ) {
          node_id = this.nodes[node_id].left_child_id
        } else {
          node_id = this.nodes[node_id].right_child_id
        }
      }
      leaf_node_ids.push(node_id)
    }
    return leaf_node_ids
  }
  PopulateChildIds(): void {
    for (let i = 1; i < this.nodes.length; i++) {
      if (this.nodes[i].is_left) {
        this.nodes[this.nodes[i].parent_id].left_child_id = i
      } else {
        this.nodes[this.nodes[i].parent_id].right_child_id = i
      }
    }
  }
  predictProba(samples: number[][]): number[][] {
    if (!this.is_built) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leaf_node_ids = this.GetLeafNodes(samples)
    let class_probabilities = []
    let current_class_probabilities = []

    for (let n_sample = 0; n_sample < leaf_node_ids.length; n_sample++) {
      current_class_probabilities = []
      let current_node_id = leaf_node_ids[n_sample]
      for (let n_class = 0; n_class < this.nodes[0].value.length; n_class++) {
        current_class_probabilities.push(
          this.nodes[current_node_id].value[n_class] /
            this.nodes[current_node_id].n_samples
        )
      }
      class_probabilities.push(current_class_probabilities)
    }
    return class_probabilities
  }
  predictClassification(samples: number[][]): int[] {
    if (!this.is_built) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leaf_node_ids = this.GetLeafNodes(samples)
    let class_predictions = []

    for (let n_sample = 0; n_sample < leaf_node_ids.length; n_sample++) {
      let current_node_id = leaf_node_ids[n_sample]
      class_predictions.push(argMax(this.nodes[current_node_id].value))
    }
    return class_predictions
  }
  predictRegression(samples: number[][]): int[] {
    if (!this.is_built) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leaf_node_ids = this.GetLeafNodes(samples)
    let class_predictions = []

    for (let n_sample = 0; n_sample < leaf_node_ids.length; n_sample++) {
      let current_node_id = leaf_node_ids[n_sample]
      class_predictions.push(this.nodes[current_node_id].value[0])
    }
    return class_predictions
  }
}

function validateX(feature_data: number[][]) {
  if (feature_data.length === 0) {
    throw new Error(
      `X can not be empty, but it has a length of 0. It is ${feature_data}.`
    )
  }
  for (let i = 0; i < feature_data.length; i++) {
    let curRow = feature_data[i]
    if (curRow.length === 0) {
      throw new Error(
        `Rows in X can not be empty, but row ${i} in X is ${curRow}.`
      )
    }
    for (let j = 0; j < curRow.length; j++) {
      if (typeof curRow[j] !== 'number' || !Number.isFinite(curRow[j])) {
        throw new Error(
          `X must contain finite non-NaN numbers, but the element at X[${i}][${j}] is ${curRow[j]}`
        )
      }
    }
  }
}

function validateY(label_data: int[]) {
  if (label_data.length === 0) {
    throw new Error(
      `y can not be empty, but it has a length of 0. It is ${label_data}.`
    )
  }
  for (let i = 0; i < label_data.length; i++) {
    let curVal = label_data[i]
    if (!Number.isSafeInteger(curVal)) {
      throw new Error(
        `Some y values are not an integer. Found ${curVal} but must be an integer only`
      )
    }
    if (curVal < 0) {
      throw new Error(
        `y values must be in the range [0, N]. This implementation expects that the labels are already normalized. We found label value ${curVal}`
      )
    }
  }
}

class DecisionTreeBase {
  splitter_!: Splitter
  stack_: NodeRecord[] = []
  min_samples_leaf: int
  max_depth_: int
  min_samples_split: int
  min_impurity_split_: number
  n_features_: int = 0
  tree_: DecisionTree
  criterion_: ImpurityMeasure
  max_features_: int
  max_features_method: 'log2_method' | 'sqrt_method' | 'all_method'
  feature_data_: number[][] = []
  label_data_: number[] = []

  constructor({
    criterion = 'gini',
    max_depth = Number.POSITIVE_INFINITY,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = -1,
    max_features_method = 'all_method',
    min_impurity_split = 0.0
  } = {}) {
    this.criterion_ = criterion as any
    this.max_depth_ = max_depth
    this.min_samples_split = min_samples_split
    this.min_samples_leaf = min_samples_leaf
    this.max_features_ = max_features
    this.max_features_method = max_features_method as any
    this.min_impurity_split_ = min_impurity_split
    this.tree_ = new DecisionTree()
  }

  public fit(
    feature_data: number[][],
    label_data: int[],
    samples_subset?: number[]
  ) {
    validateY(label_data)
    validateX(feature_data)

    this.feature_data_ = feature_data
    this.label_data_ = label_data

    let new_samples_subset = samples_subset || []

    // CheckNegativeLabels(label_data_ptr);
    this.max_features_ = SetMaxFeatures(
      this.max_features_,
      this.max_features_method,
      feature_data
    )

    this.splitter_ = new Splitter(
      feature_data,
      label_data,
      this.min_samples_leaf,
      this.criterion_,
      this.max_features_,
      new_samples_subset
    )

    // put root node on stack
    let root_node: NodeRecord = {
      start: 0,
      end: this.splitter_.sample_map_.length,
      depth: 0,
      impurity: 0,
      n_samples: this.splitter_.sample_map_.length,
      parent_id: -1,
      is_left: false
    }
    this.stack_.push(root_node)

    let is_root_node = true

    while (this.stack_.length !== 0) {
      // take next node from stack
      let current_record = this.stack_.pop() as NodeRecord
      this.splitter_.resetSampleRange(current_record.start, current_record.end)
      let current_split: Split = makeDefaultSplit()

      let is_leaf =
        !(current_record.depth < this.max_depth_) ||
        current_record.n_samples < this.min_samples_split ||
        current_record.n_samples < 2 * this.min_samples_leaf

      // evaluate abort criterion
      if (is_root_node) {
        current_record.impurity = this.splitter_.criterion_.nodeImpurity()
        is_root_node = false
      }

      // or current_record.impurity <= 0.0;
      // split unless is_leaf
      if (!is_leaf) {
        current_split = this.splitter_.splitNode()
        is_leaf =
          is_leaf ||
          !current_split.found_split ||
          current_record.impurity <= this.min_impurity_split_
      }

      let current_node: Node = {
        parent_id: current_record.parent_id,
        impurity: current_record.impurity,
        is_leaf: is_leaf,
        is_left: current_record.is_left,
        n_samples: current_record.n_samples,
        split_feature: current_split.feature,
        threshold: current_split.threshold,
        value: this.splitter_.criterion_.nodeValue().slice(),
        left_child_id: -1,
        right_child_id: -1
      }

      this.tree_.nodes.push(current_node)
      let node_id = this.tree_.nodes.length - 1

      if (!is_leaf) {
        let right_record: NodeRecord = {
          start: current_split.pos,
          end: current_record.end,
          n_samples: current_record.end - current_split.pos,
          depth: current_record.depth + 1,
          parent_id: node_id,
          is_left: false,
          impurity: current_split.impurity_right
        }

        this.stack_.push(right_record)

        let left_record: NodeRecord = {
          start: current_record.start,
          end: current_split.pos,
          n_samples: current_split.pos - current_record.start,
          depth: current_record.depth + 1,
          parent_id: node_id,
          is_left: true,
          impurity: current_split.impurity_left
        }

        this.stack_.push(left_record)
      }
    }
    this.tree_.PopulateChildIds()
    this.tree_.is_built = true
  }
}

interface DecisionTreeClassifierParams {
  criterion?: 'gini' | 'entropy'
  max_depth?: int
  min_samples_split?: number
  min_samples_leaf?: number
  max_features?: number
  min_impurity_decrease?: number
}
export class DecisionTreeClassifier extends DecisionTreeBase {
  constructor({
    criterion = 'gini',
    max_depth = Number.POSITIVE_INFINITY,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = -1,
    min_impurity_decrease = 0.0
  }: DecisionTreeClassifierParams = {}) {
    assert(
      ['gini', 'entropy'].includes(criterion as string),
      'Must pass a criterion that makes sense'
    )
    super({
      criterion,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      max_features,
      min_impurity_split: min_impurity_decrease
    })
  }
  public predict(feature_data: number[][]) {
    return this.tree_.predictClassification(feature_data)
  }

  public predictProba(feature_data: number[][]) {
    return this.tree_.predictProba(feature_data)
  }

  public score(X: number[][], y: number[]): number {
    const yPred = this.predict(X)
    return accuracyScore(y, yPred)
  }
}

interface DecisionTreeRegressorParams {
  criterion?: 'mse'
  max_depth?: int
  min_samples_split?: number
  min_samples_leaf?: number
  max_features?: number
  min_impurity_decrease?: number
}
export class DecisionTreeRegressor extends DecisionTreeBase {
  constructor({
    criterion = 'mse',
    max_depth = Number.POSITIVE_INFINITY,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = -1,
    min_impurity_decrease = 0.0
  }: DecisionTreeRegressorParams = {}) {
    assert(
      ['mse'].includes(criterion as string),
      'Must pass a criterion that makes sense'
    )
    super({
      criterion,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      max_features,
      min_impurity_split: min_impurity_decrease
    })
  }
  public predict(feature_data: number[][]) {
    return this.tree_.predictRegression(feature_data)
  }

  public score(X: number[][], y: number[]): number {
    const yPred = this.predict(X)
    return r2Score(y, yPred)
  }
}
