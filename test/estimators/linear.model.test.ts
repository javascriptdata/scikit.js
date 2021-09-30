import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { LinearRegression } from '../../dist';
import 'mocha';
import { tensorEqual } from '../utils.test';

describe('LinearRegression', function () {
  this.timeout(10000);
  it('Works on arrays (small example)', async function () {
    const lr = new LinearRegression();
    await lr.fit([[1], [2]], [2, 4]);
    assert.isTrue(tensorEqual(lr.coef_, tf.tensor2d([[2]]), 0.01));
    assert.isTrue(tensorEqual(lr.intercept_, tf.tensor1d([0]), 0.01));
  });
});
