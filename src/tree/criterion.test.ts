import { Criterion, GiniCoefficient, Entropy, SampleData } from './criterion'
import { dfd } from '../shared/globals'

describe('Criterion', function () {
  let X = [
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1]
  ]
  let y = [0, 0, 0, 1, 1, 1]
  let sample_map: SampleData[] = []
  for (let i = 0; i < X.length; i++) {
    sample_map.push({ current_feature_value: 0, sample_number: i })
  }
  it('Use the criterion (init)', async function () {
    let criterion = new Criterion('gini', y)

    criterion.init(0, 6, sample_map)
    expect(criterion.start_).toEqual(0)
    expect(criterion.end_).toEqual(6)
    expect(criterion.label_freqs_total_[0]).toEqual(3)
    expect(criterion.label_freqs_total_[1]).toEqual(3)

    expect(criterion.label_freqs_left_[0]).toEqual(0)
    expect(criterion.label_freqs_left_[1]).toEqual(0)
    expect(criterion.label_freqs_right_[0]).toEqual(0)
    expect(criterion.label_freqs_right_[1]).toEqual(0)
  }, 1000)
  it('Use the criterion (update)', async function () {
    let criterion = new Criterion('gini', y)
    criterion.init(0, 6, sample_map)
    criterion.update(3, sample_map)

    expect(criterion.pos_).toEqual(3)
    expect(criterion.label_freqs_left_[0]).toEqual(3)
    expect(criterion.label_freqs_left_[1]).toEqual(0)
    expect(criterion.label_freqs_right_[0]).toEqual(0)
    expect(criterion.label_freqs_right_[1]).toEqual(3)
  }, 1000)
  it('Use the criterion (gini)', async function () {
    let criterion = new Criterion('gini', y)

    criterion.init(0, 6, sample_map)

    expect(criterion.nodeImpurity()).toEqual(0.5)
  }, 1000)
  it('Use the criterion (entropy)', async function () {
    let criterion = new Criterion('entropy', y)
    criterion.init(0, 6, sample_map)

    expect(criterion.nodeImpurity()).toEqual(1)
  }, 1000)
  it('Use the criterion (gini update)', async function () {
    let criterion = new Criterion('gini', y)

    criterion.init(0, 6, sample_map)
    criterion.update(4, sample_map)

    expect(criterion.impurityImprovement()).toEqual(-1.5)

    let { impurity_left, impurity_right } = criterion.childrenImpurities()
    expect(impurity_left).toEqual(0.375)
    expect(impurity_right).toEqual(0)
  }, 1000)
  it('Gini coef', async function () {
    let label_freqs = [20, 80]
    let n_samples = 100
    expect(GiniCoefficient(label_freqs, n_samples)).toEqual(
      0.31999999999999995
    )
  }, 1000)
  it('Entropy coef', async function () {
    let label_freqs = [20, 80]
    let n_samples = 100
    expect(Entropy(label_freqs, n_samples)).toEqual(0.7219280948873623)
  }, 1000)
})
