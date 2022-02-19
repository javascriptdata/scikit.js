import { assert } from 'chai'
import * as metrics from './metrics'
import { describe, it } from 'mocha'

describe('Metrics', function () {
  it('accuracyScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    assert.deepEqual(metrics.accuracyScore(labels, predictions), 0.5)
  })
  it('precisionScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    assert.deepEqual(metrics.precisionScore(labels, predictions), 1)
  })
  it('recallScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    assert.deepEqual(metrics.recallScore(labels, predictions), 1)
  })
  it('meanAbsoluteError', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 0]
    assert.deepEqual(metrics.meanAbsoluteError(labels, predictions), 0.5)
  })
  it('meanSquaredError', function () {
    const labels = [1, 2, 3, 2]
    const predictions = [1, 2, 3, 0]
    assert.deepEqual(metrics.meanSquaredError(labels, predictions), 1)
  })
  it('meanSquaredLogError', function () {
    const labels = [3, 5, 2.5, 7]
    const predictions = [2.5, 5, 4, 8]
    assert.isTrue(
      Math.abs(metrics.meanSquaredLogError(labels, predictions) - 0.03973) <
        0.01
    )
  })
  it('confusionMatrix', function () {
    const labels = [2, 0, 2, 2, 0, 1]
    const predictions = [0, 0, 2, 2, 0, 2]
    const confusion = metrics.confusionMatrix(labels, predictions)
    assert.deepEqual(confusion, [
      [2, 0, 0],
      [0, 0, 1],
      [1, 0, 2]
    ])
  })
  it('hingeLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    assert.deepEqual(metrics.hingeLoss(labels, predictions), 0)
  })
  it('huberLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    assert.deepEqual(metrics.huberLoss(labels, predictions), 0.25)
  })
  it('logLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    assert.deepEqual(metrics.logLoss(labels, predictions), NaN)
  })
  it('zeroOneLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    assert.deepEqual(metrics.zeroOneLoss(labels, predictions), 0.5)
  })
  it('rocAucScore (easy)', function () {
    const labels = [0.5]
    const predictions = [1]
    assert.deepEqual(metrics.rocAucScore(labels, predictions), 0.75)
  })
  it('rocAucScore (also easy)', function () {
    const labels = [0.25, 0.75]
    const predictions = [0.5, 0.5]
    assert.deepEqual(metrics.rocAucScore(labels, predictions), 0.5)
  })
  it('empty input', function () {
    const labels: number[] = []
    const predictions: number[] = []
    assert.throws(() => metrics.accuracyScore(labels, predictions), Error)
  })
})
