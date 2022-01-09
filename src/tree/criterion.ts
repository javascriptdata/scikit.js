import { assert } from '../typesUtils'
import { int } from '../randUtils'

export type ImpurityMeasure = 'gini' | 'entropy'
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

export class Criterion {
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
    this.n_labels_ = Math.max(...label_data) + 1
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
}
