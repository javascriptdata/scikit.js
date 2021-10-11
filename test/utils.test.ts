import { assert } from 'chai'
import {
  convertToTensor1D,
  convertToTensor1D_2D,
  convertToTensor2D,
} from '../dist/utils'
import { DataFrame, Series } from 'danfojs-node'
import { tensor1d, tensor2d, tensor3d } from '@tensorflow/tfjs-core'

describe('Utility Functions', function () {
  describe('convertToTensor1D', function () {
    it('Happy path test', function () {
      const newTensor = convertToTensor1D([1, 2, 3])
      assert.isTrue(newTensor.shape.length === 1)
      newTensor.dispose()
    })
    it('Can deal with bad input', function () {
      assert.throws(() => convertToTensor1D([[1, 2, 3]] as any))
      assert.throws(() => convertToTensor1D(null as any))
      assert.throws(() => convertToTensor1D({} as any))
      assert.throws(() => convertToTensor1D('asdf' as any))
      assert.throws(() => convertToTensor1D(true as any))
      assert.throws(() => convertToTensor1D([[[1, 2, 3]]] as any))
    })
    it('Can deal with Series', function () {
      const newTensor = convertToTensor1D(new Series([1, 2, 3]))
      assert.isTrue(newTensor.shape.length === 1)
      assert.isTrue(newTensor.size === 3)
      newTensor.dispose()
    })
    it('Can deal with Tensors', function () {
      const newTensor = tensor1d([1, 2, 3])
      const newTensor2 = tensor2d([[1, 2, 3]])
      const newTensor3 = convertToTensor1D(newTensor)
      assert.isTrue(newTensor3.shape.length === 1)

      assert.throws(() => convertToTensor1D(newTensor2 as any))

      newTensor.dispose()
      newTensor2.dispose()
      newTensor3.dispose()
    })
  })

  describe('convertToTensor2D', function () {
    it('Happy path test', function () {
      const newTensor = convertToTensor2D([
        [1, 2, 3],
        [4, 5, 6],
      ])
      assert.isTrue(newTensor.shape.length === 2)
      newTensor.dispose()
    })
    it('Can deal with bad input', function () {
      assert.throws(() => convertToTensor2D([1, 2, 3] as any))
      assert.throws(() => convertToTensor2D(null as any))
      assert.throws(() => convertToTensor2D({} as any))
      assert.throws(() => convertToTensor2D('asdf' as any))
      assert.throws(() => convertToTensor2D(true as any))
      assert.throws(() => convertToTensor2D([[[1, 2, 3]]] as any))
      assert.throws(() => convertToTensor2D(new Series([1, 2, 3]) as any))
    })
    it('Can deal with Dataframes', function () {
      let df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] })
      const newTensor = convertToTensor2D(df)
      assert.isTrue(newTensor.shape.length === 2)
      newTensor.dispose()
    })
    it('Can deal with Tensors', function () {
      const newTensor = tensor1d([1, 2, 3])
      const newTensor2 = tensor2d([[1, 2, 3]])

      assert.throws(() => convertToTensor2D(newTensor as any))

      const newTensor3 = convertToTensor2D(newTensor2)
      assert.isTrue(newTensor3.shape.length === 2)

      newTensor.dispose()
      newTensor2.dispose()
      newTensor3.dispose()
    })
  })
  describe('convertToTensor1D_2DTensor', function () {
    it('Happy path test', function () {
      const newTensor = convertToTensor1D_2D([
        [1, 2, 3],
        [4, 5, 6],
      ])
      assert.isTrue(newTensor.shape.length === 2)
      const newTensor2 = convertToTensor1D_2D([1, 2, 3])
      assert.isTrue(newTensor2.shape.length === 1)
      newTensor.dispose()
      newTensor2.dispose()
    })
    it('Can deal with bad input', function () {
      assert.throws(() => convertToTensor1D_2D(null as any))
      assert.throws(() => convertToTensor1D_2D({} as any))
      assert.throws(() => convertToTensor1D_2D('asdf' as any))
      assert.throws(() => convertToTensor1D_2D(true as any))
      assert.throws(() => convertToTensor1D_2D([[[1, 2, 3]]] as any))
    })
    it('Can deal with Series', function () {
      const newTensor = convertToTensor1D_2D(new Series([1, 2, 3]))
      assert.isTrue(newTensor.shape.length === 1)
      assert.isTrue(newTensor.size === 3)
      newTensor.dispose()
    })
    it('Can deal with Dataframes', function () {
      let df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] })
      const newTensor = convertToTensor1D_2D(df)
      assert.isTrue(newTensor.shape.length === 2)
      newTensor.dispose()
    })
    it('Can deal with Tensors', function () {
      const newTensor = tensor1d([1, 2, 3])
      const newTensor2 = tensor2d([[1, 2, 3]])
      const newTensor3 = tensor3d([[[1, 2, 3]]])

      assert.throws(() => convertToTensor1D_2D(newTensor3 as any))

      const convTensor = convertToTensor1D_2D(newTensor)
      assert.isTrue(newTensor.shape.length === 1)

      const convTensor2 = convertToTensor1D_2D(newTensor2)
      assert.isTrue(newTensor2.shape.length === 2)

      newTensor.dispose()
      newTensor2.dispose()
      newTensor3.dispose()
      convTensor.dispose()
      convTensor2.dispose()
    })
  })
  // it('A simple test that to make sure we can create 1D Tensor', function () {
  //   const newTensor = convertToTensor1D(null)
  //   assert.isTrue(newTensor.shape.length === 1)
  //   newTensor.dispose()
  // })
})
