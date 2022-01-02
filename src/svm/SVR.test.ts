import { SVR } from './SVR'


describe('SVR', function () {
  it('Works on arrays (small example)', async function () {
    const lr = new SVR()
    await lr.fit(
      [
        [1, 1, 1],
        [-2, -2, -2]
      ],
      [-1, 1]
    )
    const predict = (
      await lr.predict([
        [1, 1, 1],
        [-2, -2, -2]
      ])
    ).arraySync()
    const ans = [-1, 1]
    predict.forEach((it, idx) => {
      expect(Math.abs(it - ans[idx])).toBeLessThanOrEqual(5e-1)
    })
  }, 10000)
})
