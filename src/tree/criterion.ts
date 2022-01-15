import { assert } from '../typesUtils'
import { int } from '../randUtils'

export type ImpurityMeasure = 'gini' | 'entropy' | 'mse'
export interface SampleData {
  sample_number: int
  current_feature_value: number
}

export function GiniCoefficient(label_freqs: int[], n_samples_: int) {
  let freq_squares = 0
  for (let i = 0; i < label_freqs.length; i++) {
    freq_squares += label_freqs[i] * label_freqs[i]
  }
  return 1 - freq_squares / (n_samples_ * n_samples_)
}

export function Entropy(label_freqs: int[], n_samples: int) {
  let entropy = 0
  for (let i = 0; i < label_freqs.length; i++) {
    let label_frequency = label_freqs[i]
    if (label_frequency > 0) {
      label_frequency /= n_samples
      entropy -= label_frequency * Math.log2(label_frequency)
    }
  }
  return entropy
}

export function MSE(y_squared_sum: number, y_sum: number, n_samples: int) {
  let y_bar = y_sum / n_samples
  let val = y_squared_sum / n_samples - y_bar * y_bar
  return val
}

function arrayMax(labels: int[]) {
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > max) {
      max = labels[i]
    }
  }
  return max
}

export class ClassificationCriterion {
  label_data_: int[]
  impurity_measure_: ImpurityMeasure
  impurity_fn_: (label_freqs: int[], n_samples: int) => number
  start_: int = 0
  end_: int = 0
  pos_: int = 0
  n_labels_: int
  label_freqs_total_: int[] = []
  label_freqs_left_: int[] = []
  label_freqs_right_: int[] = []
  n_samples_: int = 0
  n_samples_left_: int = 0
  n_samples_right_: int = 0

  constructor(impurity_measure: ImpurityMeasure, label_data: number[]) {
    assert(
      ['gini', 'entropy'].includes(impurity_measure),
      'Unkown impurity measure. Only supports gini, and entropy'
    )

    this.impurity_measure_ = impurity_measure
    if (this.impurity_measure_ === 'gini') {
      this.impurity_fn_ = GiniCoefficient
    } else {
      this.impurity_fn_ = Entropy
    }
    // This assumes that the labels are 0,1,2,...,(n-1)
    this.n_labels_ = arrayMax(label_data) + 1
    this.label_data_ = label_data
    this.label_freqs_total_ = new Array(this.n_labels_).fill(0)
    this.label_freqs_left_ = new Array(this.n_labels_).fill(0)
    this.label_freqs_right_ = new Array(this.n_labels_).fill(0)
  }

  init(start: int, end: int, sample_map: SampleData[]) {
    this.start_ = start
    this.end_ = end
    this.n_samples_ = end - start
    this.label_freqs_total_ = this.label_freqs_total_.fill(0)
    this.label_freqs_left_ = this.label_freqs_left_.fill(0)
    this.label_freqs_right_ = this.label_freqs_right_.fill(0)

    for (let i = start; i < end; i++) {
      let sampleNumber = sample_map[i].sample_number
      this.label_freqs_total_[this.label_data_[sampleNumber]] += 1
    }
  }

  reset() {
    this.pos_ = this.start_
    this.label_freqs_left_ = this.label_freqs_left_.fill(0)
    this.label_freqs_right_ = this.label_freqs_right_.fill(0)
  }

  update(new_pos: int, sample_map: SampleData[]) {
    for (let i = this.pos_; i < new_pos; i++) {
      // This assumes that the labels take values 0,..., n_labels - 1
      let sampleNumber = sample_map[i].sample_number
      this.label_freqs_left_[this.label_data_[sampleNumber]] += 1
    }

    // calculate label_freqs_right_
    for (let i = 0; i < this.label_freqs_total_.length; i++) {
      this.label_freqs_right_[i] =
        this.label_freqs_total_[i] - this.label_freqs_left_[i]
    }

    this.pos_ = new_pos
    this.n_samples_left_ = this.pos_ - this.start_
    this.n_samples_right_ = this.end_ - this.pos_
  }

  childrenImpurities() {
    return {
      impurity_left: this.impurity_fn_(
        this.label_freqs_left_,
        this.n_samples_left_
      ),
      impurity_right: this.impurity_fn_(
        this.label_freqs_right_,
        this.n_samples_right_
      )
    }
  }

  impurityImprovement() {
    let { impurity_left, impurity_right } = this.childrenImpurities()

    return (
      -this.n_samples_left_ * impurity_left -
      this.n_samples_right_ * impurity_right
    )
  }

  nodeImpurity() {
    return this.impurity_fn_(this.label_freqs_total_, this.n_samples_)
  }

  nodeValue() {
    return this.label_freqs_total_
  }
}

export class RegressionCriterion {
  label_data_: number[]
  impurity_measure_: 'mse'
  impurity_fn_: (
    y_squared_sum: number,
    y_sum: number,
    n_samples: int
  ) => number
  start_: int = 0
  end_: int = 0
  pos_: int = 0
  squared_sum = 0
  squared_sum_left = 0
  squared_sum_right = 0
  sum_total = 0
  sum_total_left = 0
  sum_total_right = 0
  n_samples_: int = 0
  n_samples_left_: int = 0
  n_samples_right_: int = 0

  constructor(impurity_measure: 'mse', label_data: number[]) {
    assert(
      ['mse'].includes(impurity_measure),
      'Unkown impurity measure. Only supports mse'
    )

    // Support MAE one day
    this.impurity_measure_ = impurity_measure
    this.impurity_fn_ = MSE
    this.label_data_ = label_data
  }

  init(start: int, end: int, sample_map: SampleData[]) {
    this.sum_total = 0
    this.squared_sum = 0
    this.start_ = start
    this.end_ = end
    this.n_samples_ = end - start

    for (let i = start; i < end; i++) {
      let sampleNumber = sample_map[i].sample_number
      let y_value = this.label_data_[sampleNumber]
      this.sum_total += y_value
      this.squared_sum += y_value * y_value
    }
  }

  reset() {
    this.pos_ = this.start_
    this.squared_sum_left = 0
    this.sum_total_left = 0
    this.squared_sum_right = 0
    this.sum_total_right = 0
  }

  update(new_pos: int, sample_map: SampleData[]) {
    for (let i = this.pos_; i < new_pos; i++) {
      // This assumes that the labels take values 0,..., n_labels - 1
      let sampleNumber = sample_map[i].sample_number
      let y_value = this.label_data_[sampleNumber]
      this.sum_total_left += y_value
      this.squared_sum_left += y_value * y_value
    }

    // calculate label_freqs_right_
    this.sum_total_right = this.sum_total - this.sum_total_left
    this.squared_sum_right = this.squared_sum - this.squared_sum_left

    this.pos_ = new_pos
    this.n_samples_left_ = this.pos_ - this.start_
    this.n_samples_right_ = this.end_ - this.pos_
  }

  childrenImpurities() {
    return {
      impurity_left: this.impurity_fn_(
        this.squared_sum_left,
        this.sum_total_left,
        this.n_samples_left_
      ),
      impurity_right: this.impurity_fn_(
        this.squared_sum_right,
        this.sum_total_right,
        this.n_samples_right_
      )
    }
  }

  impurityImprovement() {
    let { impurity_left, impurity_right } = this.childrenImpurities()

    return (
      -this.n_samples_left_ * impurity_left -
      this.n_samples_right_ * impurity_right
    )
  }

  nodeImpurity() {
    return this.impurity_fn_(this.squared_sum, this.sum_total, this.n_samples_)
  }

  nodeValue() {
    return [this.sum_total / this.n_samples_]
  }
}
