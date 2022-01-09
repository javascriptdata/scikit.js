import { ImpurityMeasure } from './criterion'
import { Splitter } from './splitter'
import { deepCopy } from './utils'
import { int } from '../randUtils'
import { ClassifierMixin } from '../mixins'

interface NodeRecord {
  start: int
  end: int
  n_samples: int
  depth: int
  parent_id: int
  is_left: boolean
  impurity: number
  value: int[]
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
  AddNode(node: Node) {
    this.nodes.push(node)
    return this.nodes.length - 1
  }
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
  predict(samples: number[][]): int[] {
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
      this.nodes[current_node_id].value
    }
    return class_predictions
  }
}

export class DecisionTreeClassifier extends ClassifierMixin {
  splitter_?: Splitter
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
    super()
    this.criterion_ = criterion as any
    this.max_depth_ = max_depth
    this.min_samples_split = min_samples_split
    this.min_samples_leaf = min_samples_leaf
    this.max_features_ = max_features
    this.max_features_method = max_features_method as any
    this.min_impurity_split_ = min_impurity_split
    this.tree_ = new DecisionTree()
  }

  predict(feature_data: number[][]) {
    return this.tree_.predict(feature_data)
  }

  predictProba(feature_data: number[][]) {
    return this.tree_.predictProba(feature_data)
  }
  fit(feature_data: number[][], label_data: int[], samples_subset?: number[]) {
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
      is_left: false,
      value: []
    }
    this.stack_.push(root_node)

    let is_root_node = true

    while (this.stack_.length !== 0) {
      // take next node from stack
      let current_record = this.stack_.pop() as NodeRecord
      let current_split = null
      // evaluate abort criterion
      let is_leaf =
        !(current_record.depth < this.max_depth_) ||
        current_record.n_samples < this.min_samples_split ||
        current_record.n_samples < 2 * this.min_samples_leaf
      // or current_record.impurity <= 0.0;
      // split unless is_leaf
      if (!is_leaf) {
        this.splitter_.resetSampleRange(
          current_record.start,
          current_record.end
        )
        current_split = this.splitter_.splitNode()
      }

      if (is_root_node) {
        this.splitter_.resetSampleRange(
          current_record.start,
          current_record.end
        )
        current_split = this.splitter_.splitNode()
        current_record.impurity = this.splitter_.criterion_.nodeImpurity()
        current_record.value = deepCopy(
          this.splitter_.criterion_.label_freqs_total_
        )
        is_root_node = false
      }

      is_leaf =
        is_leaf ||
        !current_split?.found_split ||
        current_record.impurity <= this.min_impurity_split_

      let current_node: Node = {
        parent_id: current_record.parent_id,
        impurity: current_record.impurity,
        is_leaf: is_leaf,
        is_left: current_record.is_left,
        n_samples: current_record.n_samples,
        split_feature: current_split?.feature as number,
        threshold: current_split?.threshold as number,
        value: current_record.value,
        left_child_id: -1,
        right_child_id: -1
      }

      let node_id = this.tree_.AddNode(current_node)

      if (!is_leaf) {
        let right_record: NodeRecord = {
          start: current_split?.pos as number,
          end: current_record.end,
          n_samples: current_record.end - (current_split?.pos as number),
          depth: current_record.depth + 1,
          parent_id: node_id,
          is_left: false,
          impurity: current_split?.impurity_right as number,
          value: current_split?.right_value as int[]
        }

        this.stack_.push(right_record)

        let left_record: NodeRecord = {
          start: current_record.start,
          end: current_split?.pos as number,
          n_samples: (current_split?.pos as number) - current_record.start,
          depth: current_record.depth + 1,
          parent_id: node_id,
          is_left: true,
          impurity: current_split?.impurity_left as number,
          value: current_split?.left_value as int[]
        }

        this.stack_.push(left_record)
      }
    }
    this.tree_.PopulateChildIds()
    this.tree_.is_built = true
  }
}
