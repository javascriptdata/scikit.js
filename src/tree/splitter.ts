import {
  ClassificationCriterion,
  RegressionCriterion,
  ImpurityMeasure,
  SampleData
} from './criterion'
import { shuffle } from 'lodash'
import { quickSort } from './utils'
import { int } from '../randUtils'

export class Split {
  feature: int = 0
  threshold = 0
  pos = -1
  impurity_left: number = Number.POSITIVE_INFINITY
  impurity_right: number = Number.POSITIVE_INFINITY
  left_value: int[] = []
  right_value: int[] = []
  found_split = false
}

export class Splitter {
  kMinSplitDiff_: number
  feature_data_: number[][]
  label_data_: int[]
  criterion_: ClassificationCriterion | RegressionCriterion
  start_: int
  end_: int
  min_samples_leaf_: int
  max_features_: int
  feature_order_: int[]
  shuffle_features_: boolean

  sample_map_: SampleData[]
  n_samples_total_: int
  n_features_: int
  constructor(
    feature_data: number[][],
    label_data: int[],
    min_samples_leaf: int,
    impurity_measure: ImpurityMeasure,
    max_features: int,
    samples_subset: int[] = []
  ) {
    this.feature_data_ = feature_data
    this.label_data_ = label_data
    this.n_features_ = feature_data[0].length
    this.min_samples_leaf_ = min_samples_leaf
    this.max_features_ = max_features
    this.shuffle_features_ = max_features < this.n_features_
    this.sample_map_ = []
    this.start_ = 0
    this.end_ = 0
    this.kMinSplitDiff_ = 1e-8
    if (samples_subset.length === 0) {
      this.n_samples_total_ = feature_data.length
      for (let i = 0; i < this.n_samples_total_; i++) {
        this.sample_map_.push({ current_feature_value: 0, sample_number: i })
      }
    } else {
      this.n_samples_total_ = samples_subset.length
      for (let i = 0; i < this.n_samples_total_; i++) {
        this.sample_map_.push({
          current_feature_value: 0,
          sample_number: samples_subset[i]
        })
      }
    }
    if (impurity_measure === 'mse') {
      this.criterion_ = new RegressionCriterion(impurity_measure, label_data)
    } else {
      this.criterion_ = new ClassificationCriterion(
        impurity_measure,
        label_data
      )
    }
    this.feature_order_ = []
    for (let i = 0; i < this.n_features_; i++) {
      this.feature_order_.push(i)
    }
    this.resetSampleRange(0, this.n_samples_total_)
  }

  resetSampleRange(start: int, end: int) {
    this.start_ = start
    this.end_ = end
    this.criterion_.init(start, end, this.sample_map_)
  }

  splitNode() {
    let current_split = new Split()
    let best_split = new Split()
    let current_impurity_improvement = Number.NEGATIVE_INFINITY
    let best_impurity_improvement = Number.NEGATIVE_INFINITY
    let current_feature_num = 0
    let current_feature = 0
    current_split.found_split = false
    if (this.shuffle_features_) {
      this.feature_order_ = shuffle(this.feature_order_)
    }

    while (current_feature_num < this.max_features_) {
      current_feature = this.feature_order_[current_feature_num]

      // Copies feature data into sample map
      for (let i = this.start_; i < this.end_; i++) {
        this.sample_map_[i].current_feature_value =
          this.feature_data_[this.sample_map_[i].sample_number][
            current_feature
          ]
      }
      this.criterion_.reset()

      this.sample_map_ = quickSort(
        this.sample_map_,
        this.start_,
        this.end_ - 1,
        'current_feature_value'
      )

      // If this feature value is constant, then skip it.
      if (
        this.sample_map_[this.start_].current_feature_value ===
        this.sample_map_[this.end_ - 1].current_feature_value
      ) {
        current_feature_num += 1
        continue
      }
      let pos = this.start_ + 1
      // Loop over all split points
      while (pos < this.end_) {
        // Skip split points where the features are equal because
        // you can't "slice" there
        while (
          pos < this.end_ &&
          this.sample_map_[pos].current_feature_value <=
            this.sample_map_[pos - 1].current_feature_value +
              this.kMinSplitDiff_
        ) {
          pos++
        }
        if (pos === this.end_) {
          pos++
          continue
        }
        // Check if split would lead to less than min_samples_leaf samples
        if (
          !(
            pos - this.start_ < this.min_samples_leaf_ ||
            this.end_ - pos < this.min_samples_leaf_
          )
        ) {
          current_split.pos = pos
          this.criterion_.update(current_split.pos, this.sample_map_)
          current_impurity_improvement = this.criterion_.impurityImprovement()
          if (current_impurity_improvement > best_impurity_improvement) {
            best_impurity_improvement = current_impurity_improvement
            current_split.found_split = true
            current_split.feature = current_feature
            current_split.threshold =
              (this.sample_map_[pos - 1].current_feature_value +
                this.sample_map_[pos].current_feature_value) /
              2.0

            best_split = Object.assign({}, current_split)
            if (this.criterion_ instanceof ClassificationCriterion) {
              best_split.left_value = this.criterion_.label_freqs_left_.slice()

              best_split.right_value =
                this.criterion_.label_freqs_right_.slice()
            } else {
              best_split.left_value = this.criterion_.sum_total_left as any
              best_split.right_value = this.criterion_.sum_total_right as any
            }
          }
        }

        // increment the position
        pos += 1
      }
      // increment the feature that we are looking at
      current_feature_num += 1
    }

    if (current_split.found_split) {
      if (best_split.pos < this.end_) {
        if (current_feature !== best_split.feature) {
          let left_pos = this.start_
          let right_pos = this.end_
          let tmp = 0
          while (left_pos < right_pos) {
            if (
              this.feature_data_[this.sample_map_[left_pos].sample_number][
                best_split.feature
              ] <= best_split.threshold
            ) {
              left_pos += 1
            } else {
              right_pos -= 1
              tmp = this.sample_map_[left_pos].sample_number
              this.sample_map_[left_pos].sample_number =
                this.sample_map_[right_pos].sample_number
              this.sample_map_[right_pos].sample_number = tmp
            }
          }
        }
      }

      this.criterion_.reset()
      this.criterion_.update(best_split.pos, this.sample_map_)
      let { impurity_left, impurity_right } =
        this.criterion_.childrenImpurities()

      best_split.impurity_left = impurity_left
      best_split.impurity_right = impurity_right

      return best_split
    } else {
      // passing back split.found_split = false
      return current_split
    }
  }
}
