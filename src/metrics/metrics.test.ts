import * as metrics from './metrics'


describe('Metrics', function () {
  it('accuracyScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    expect(metrics.accuracyScore(labels, predictions)).toEqual(0.5)
  })
  it('precisionScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    expect(metrics.precisionScore(labels, predictions)).toEqual(1)
  })
  it('recallScore', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 4]
    expect(metrics.recallScore(labels, predictions)).toEqual(1)
  })
  it('meanAbsoluteError', function () {
    const labels = [1, 2, 3, 1]
    const predictions = [1, 2, 4, 0]
    expect(metrics.meanAbsoluteError(labels, predictions)).toEqual(0.5)
  })
  it('meanSquaredError', function () {
    const labels = [1, 2, 3, 2]
    const predictions = [1, 2, 3, 0]
    expect(metrics.meanSquaredError(labels, predictions)).toEqual(1)
  })
  it('meanSquaredLogError', function () {
    const labels = [3, 5, 2.5, 7]
    const predictions = [2.5, 5, 4, 8]
    expect(Math.abs(metrics.meanSquaredLogError(labels, predictions) - 0.03973) <
      0.01).toBe(true)
  })
  it('confusionMatrix', function () {
    const labels = [2, 0, 2, 2, 0, 1]
    const predictions = [0, 0, 2, 2, 0, 2]
    const confusion = metrics.confusionMatrix(labels, predictions)
    expect(confusion).toEqual([
      [2, 0, 0],
      [0, 0, 1],
      [1, 0, 2]
    ])
  })
  it('hingeLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    expect(metrics.hingeLoss(labels, predictions)).toEqual(0)
  })
  it('huberLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    expect(metrics.huberLoss(labels, predictions)).toEqual(0.25)
  })
  it('logLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    expect(metrics.logLoss(labels, predictions)).toEqual(NaN)
  })
  it('zeroOneLoss', function () {
    const labels = [3, 5, 4, 7]
    const predictions = [4, 5, 4, 8]
    expect(metrics.zeroOneLoss(labels, predictions)).toEqual(0.5)
  })
  it('rocAucScore (easy)', function () {
    const labels = [0.5]
    const predictions = [1]
    expect(metrics.rocAucScore(labels, predictions)).toEqual(0.75)
  })
  it('rocAucScore (also easy)', function () {
    const labels = [0.25, 0.75]
    const predictions = [0.5, 0.5]
    expect(metrics.rocAucScore(labels, predictions)).toEqual(0.5)
  })
  it('empty input', function () {
    const labels: number[] = []
    const predictions: number[] = []
    expect(() => metrics.accuracyScore(labels, predictions)).toThrow()
  })
})
