import * as tf from '@tensorflow/tfjs'
import { KMeans, setBackend, fromJSON } from '../index'
setBackend(tf)
// Next steps: Improve on kmeans cluster testing
describe('KMeans', () => {
  const X = [
    [1, 2],
    [1, 4],
    [1, 0],
    [4, 2],
    [4, 4],
    [4, 0]
  ]

  it('should fit vector1 + k=2 should return centroids of size 2 and clusters of size 2', () => {
    const expecterdCluster = {
      centroids: [
        [2.5, 1],
        [2.5, 4]
      ],
      clusters: [
        [
          [1, 2],
          [1, 0],
          [4, 2],
          [4, 0]
        ],
        [
          [1, 4],
          [4, 4]
        ]
      ],
      k: 2
    }
    const kmean = new KMeans({ nClusters: 2, randomState: 0 })
    kmean.fit(X)
    expect(expecterdCluster.centroids).toEqual(
      kmean.clusterCenters.arraySync()
    )
  })

  it('should save kmeans model', async () => {
    const expectedResult = {
      name: 'KMeans',
      nClusters: 2,
      init: 'random',
      maxIter: 300,
      tol: 0.0001,
      randomState: 0,
      nInit: 10,
      clusterCenters: {
        name: 'Tensor',
        value: [
          [2.5, 1],
          [2.5, 4]
        ]
      }
    }
    const kmean = new KMeans({ nClusters: 2, randomState: 0 })
    kmean.fit(X)
    delete kmean.tf
    const ksave = await kmean.toObject()

    expect(expectedResult).toEqual(ksave)
  })

  it('should load serialized kmeans model', async () => {
    const centroids = [
      [2.5, 1],
      [2.5, 4]
    ]
    const kmean = new KMeans({ nClusters: 2, randomState: 0 })
    kmean.fit(X)
    const ksave = await kmean.toJSON()
    const ksaveModel = await fromJSON(ksave)
    expect(centroids).toEqual(ksaveModel.clusterCenters.arraySync())
  })

  // it('should fit vector1 + k=3 size 3 and clusters of size 2', () => {
  //   const expectedCluster = {
  //     centroids: [
  //       [2.5, 2],
  //       [2.5, 4],
  //       [2.5, 0]
  //     ],
  //     clusters: [
  //       [
  //         [1, 2],
  //         [4, 2]
  //       ],
  //       [
  //         [1, 4],
  //         [4, 4]
  //       ],
  //       [
  //         [1, 0],
  //         [4, 0]
  //       ]
  //     ],
  //     k: 3
  //   }
  //   const kmean = new KMeans({ k: 3 })
  //   kmean.fit(vector1)
  //   expect(_.isEqual(expectedCluster, kmean.toJSON())).toBe(true)
  // })

  // it('should predict [2, 1] from predVector1 prediction', () => {
  //   const expectedResult = [2, 1]
  //   const kmean = new KMeans({ k: 3 })
  //   kmean.fit(vector1)
  //   const pred = kmean.predict(predVector1)
  //   expect(_.isEqual(pred, expectedResult)).toBe(true)
  // })

  // it('should predict [ 0, 0 ] from predVector2 prediction', () => {
  //   const expectedResult = [0, 0]
  //   const kmean = new KMeans({ k: 3 })
  //   kmean.fit(vector1)
  //   const pred = kmean.predict(predVector2)
  //   expect(_.isEqual(pred, expectedResult)).toBe(true)
  // })

  // it('should predict [ 0, 0 ] from predVector2 with X: vector1', () => {
  //   const expectedResult = [0, 0]
  //   const kmean = new KMeans({ k: 2 })
  //   kmean.fit(vector1)
  //   const pred = kmean.predict(predVector2)
  //   expect(_.isEqual(expectedResult, pred)).toBe(true)
  // })

  // it('should predict the same after reloading the model', () => {
  //   const expectedResult = [0, 0]
  //   const kmean = new KMeans({ k: 2 })
  //   kmean.fit(vector1)
  //   const pred1 = kmean.predict(predVector2)
  //   expect(_.isEqual(expectedResult, pred1)).toBe(true)

  //   const checkpoint = kmean.toJSON()

  //   const kmean2 = new KMeans()
  //   kmean2.fromJSON(checkpoint)
  //   const pred2 = kmean.predict(predVector2)
  //   expect(_.isEqual(expectedResult, pred2)).toBe(true)
  // })

  // it('should not fit none 2D matrix', () => {
  //   const kmean = new KMeans({ k: 2 })

  //   try {
  //     kmean.fit([1, 2] as any)
  //   } catch (err) {
  //     expect(err).toBeInstanceOf(Validation2DMatrixError)
  //   }
  //   try {
  //     kmean.fit(null as any)
  //   } catch (err) {
  //     expect(err).toBeInstanceOf(ValidationError)
  //   }
  //   // Next steps: implement datatype check to the validation method
  //   // expect(() => kmean.fit([["aa", "bb"]])).toThrow('The matrix is not 2D shaped: [1,2] of [2]');
  // })
})
