import { LogisticRegression, setBackend, fromJSON } from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('LogisticRegression', function () {
  it('Works on arrays (small example)', async function () {
    const lr = new LogisticRegression()

    await lr.fit([[1], [2]], [0, 1])
    expect(lr.predict([[1], [2]]).arraySync()).toEqual([0, 1])
  }, 30000)
  it('Test of the function used with 2 classes', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4]
    ]
    let y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 4.5],
      [2, 3.5],
      [1.5, 5]
    ]

    let logreg = new LogisticRegression({ penalty: 'none' })
    await logreg.fit(X, y)
    let probabilities = logreg.predictProba(X)
    expect(probabilities instanceof tf.Tensor).toBe(true)
    let results = logreg.predict(Xtest) // compute results of the training set
    expect(results.arraySync()).toEqual([0, 0, 0, 1, 1, 1])
    expect(logreg.score(X, y) > 0.5).toBe(true)
  }, 30000)
  it('Test of the function used with 2 classes (one hot)', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4]
    ]
    let y = [
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1]
    ]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 4.5],
      [2, 3.5],
      [1.5, 5]
    ]

    let logreg = new LogisticRegression({ penalty: 'none' })
    await logreg.fit(X, y)
    let probabilities = logreg.predictProba(X)
    expect(probabilities instanceof tf.Tensor).toBe(true)
    let results = logreg.predict(Xtest) // compute results of the training set
    expect(results.arraySync()).toEqual([
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 1],
      [0, 1],
      [0, 1]
    ])
    expect(logreg.score(X, y) > 0.5).toBe(true)
  }, 30000)
  it('Test of the prediction with 3 classes', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11]
    ]
    let y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5]
    ]

    let logreg = new LogisticRegression({ penalty: 'none' })
    await logreg.fit(X, y)
    let finalResults = logreg.predict(Xtest)
    expect(finalResults.arraySync()).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2])
  }, 30000)

  it('Should save and load trained model', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11]
    ]
    let y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5]
    ]

    let logreg = new LogisticRegression({ penalty: 'l2' })
    await logreg.fit(X, y)

    const serializeModel = await logreg.toJSON()
    const newModel = await fromJSON(serializeModel)
    const newModelResult = newModel.predict(Xtest)

    expect(newModelResult.arraySync()).toEqual([0, 0, 0, 0, 0, 0, 2, 2, 2])
  }, 30000)
})
