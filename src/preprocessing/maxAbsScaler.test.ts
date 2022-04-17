import { MaxAbsScaler } from './maxAbsScaler'
import * as dfd from 'danfojs-node'
import { tensor2d } from '@tensorflow/tfjs-core'
import { arrayEqual } from '../utils'

describe('MaxAbsScaler', function () {
  it('Standardize values in a DataFrame using a MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()

    const data = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]

    const expected = [
      [-1, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [1, 1]
    ]

    scaler.fit(new dfd.DataFrame(data))
    const resultDf = new dfd.DataFrame(
      scaler.transform(new dfd.DataFrame(data))
    )
    expect(resultDf.values).toEqual(expected)
    expect(scaler.transform([[2, 5]]).arraySync()).toEqual([[2, 0.5]])
  })
  it('fitTransform using a MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()
    const data = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]

    const expected = [
      [-1, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [1, 1]
    ]
    const resultDf = new dfd.DataFrame(
      scaler.fitTransform(new dfd.DataFrame(data))
    )

    expect(resultDf.values).toEqual(expected)
  })
  it('InverseTransform with MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()
    scaler.fit(tensor2d([1, 5, 10, 10, 5], [5, 1])) // scaling factor is 10
    const resultTransform = scaler.transform(
      tensor2d([10, 5, 50, 100, 50], [5, 1])
    )
    expect(resultTransform.arraySync().flat()).toEqual([1, 0.5, 5, 10, 5])

    const resultInverse = scaler.inverseTransform(
      tensor2d([0.1, 0.5, 1, 1, 0.5], [5, 1])
    )
    expect([1, 5, 10, 10, 5]).toEqual(resultInverse.arraySync().flat())
  })
  it('Handles pathological examples with constant features with MaxAbsScaler', function () {
    const data = [[0, 0, 0, 0]]
    const scaler = new MaxAbsScaler()
    scaler.fit(data)
    expect(scaler.transform([[0, 0, 0, 0]]).arraySync()).toEqual([
      [0, 0, 0, 0]
    ])

    expect(scaler.transform([[10, 10, -10, 10]]).arraySync()).toEqual([
      [10, 10, -10, 10]
    ])
  })
  it('Errors when you pass garbage input into a MaxAbsScaler', function () {
    const data = 4
    const scaler = new MaxAbsScaler()
    expect(() => scaler.fit(data as any)).toThrow()
  })
  it('Gracefully handles Nan as inputs MaxAbsScaler', function () {
    const data = tensor2d([4, 4, 'whoops', 4, -4] as any, [5, 1])
    const scaler = new MaxAbsScaler()
    scaler.fit(data as any)
    expect(scaler.transform(data as any).arraySync()).toEqual([
      [1],
      [1],
      [NaN],
      [1],
      [-1]
    ])
  })
  it('keeps track of variables', function () {
    let myDf = new dfd.DataFrame({ a: [1, 2, 3, 4], b: [5, 6, 7, 8] })
    let scaler = new MaxAbsScaler()
    scaler.fit(myDf)
    expect(scaler.nSamplesSeen).toEqual(4)
    expect(scaler.nFeaturesIn).toEqual(2)
    expect(scaler.featureNamesIn).toEqual(['a', 'b'])
  })
  it('test_maxabs_scaler_zero_variance_features', function () {
    // Check MaxAbsScaler on toy data with zero variance features
    let X = [
      [0.0, 1.0, +0.5],
      [0.0, 1.0, -0.3],
      [0.0, 1.0, +1.5],
      [0.0, 0.0, +0.0]
    ]

    let scaler = new MaxAbsScaler()
    let X_trans = scaler.fitTransform(X).arraySync()
    let X_expected = [
      [0.0, 1.0, 1.0 / 3.0],
      [0.0, 1.0, -0.2],
      [0.0, 1.0, 1.0],
      [0.0, 0.0, 0.0]
    ]
    expect(arrayEqual(X_trans, X_expected, 0.01)).toBe(true)
    let X_trans_inv = scaler.inverseTransform(X_trans).arraySync()
    expect(arrayEqual(X, X_trans_inv, 0.01)).toBe(true)

    // make sure new data gets transformed correctly
    let X_new = [
      [+0.0, 2.0, 0.5],
      [-1.0, 1.0, 0.0],
      [+0.0, 1.0, 1.5]
    ]
    let X_trans_new = scaler.transform(X_new).arraySync()
    let X_expected_new = [
      [+0.0, 2.0, 1.0 / 3.0],
      [-1.0, 1.0, 0.0],
      [+0.0, 1.0, 1.0]
    ]

    expect(arrayEqual(X_trans_new, X_expected_new, 0.01)).toBe(true)
  })
  /* Streaming test
  def test_maxabs_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d[:100, :]
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = MaxAbsScaler().fit(X)

        scaler_incr = MaxAbsScaler()
        scaler_incr_csr = MaxAbsScaler()
        scaler_incr_csc = MaxAbsScaler()
        for batch in gen_batches(n, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            X_csr = sparse.csr_matrix(X[batch])
            scaler_incr_csr = scaler_incr_csr.partial_fit(X_csr)
            X_csc = sparse.csc_matrix(X[batch])
            scaler_incr_csc = scaler_incr_csc.partial_fit(X_csc)

        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csc.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csc.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csc.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_batch = MaxAbsScaler().fit(X[batch0])
        scaler_incr = MaxAbsScaler().partial_fit(X[batch0])

        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # Test std until the end of partial fits, and
        scaler_batch = MaxAbsScaler().fit(X)
        scaler_incr = MaxAbsScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )
  */
})
