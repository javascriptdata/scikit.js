import '@tensorflow/tfjs-backend-webgl'
import { assert } from 'chai'
import { SVR } from './SVR'
import { describe, it } from 'mocha'

describe('SVR', function () {
  this.timeout(30000)
  it('Works on arrays (small example)', async function () {
    const lr = new SVR()

    await lr.fit([[1], [2]], [0, 1])
    const predict = (await lr.predict([[1], [2]])).arraySync();
    const ans = [1, 2];
    predict.forEach((it, idx) => {
      assert.closeTo(it, ans[idx], 1.0);
    });
  })
})
