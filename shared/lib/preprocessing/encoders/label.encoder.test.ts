import { assert } from 'chai'
import { LabelEncoder } from './label.encoder'
import { dfd } from '../../../globals'
import 'mocha'

describe('LabelEncoder', function () {
  it('LabelEncoder works for Series', function () {
    const sf = new dfd.Series([1, 2, 2, 6])
    const scaler = new LabelEncoder()
    scaler.fit(sf)
    const expected = [0, 1, 1, 2]
    assert.deepEqual(scaler.transform(sf).arraySync(), expected)
  })
  it('LabelEncoder works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const expected = [0, 1, 1, 2, 3, 3]
    assert.deepEqual(scaler.transform(sf as any).arraySync(), expected)
  })
  it('fitTransform works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    const result = scaler.fitTransform(sf)
    const expected = [0, 1, 1, 2, 3, 3]
    assert.deepEqual(result.arraySync(), expected)
  })
  it('inverseTransform works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const result = scaler.inverseTransform([0, 1, 1, 2, 3, 3])
    assert.deepEqual(result, [1, 2, 2, 'boy', 'git', 'git'])
  })
  it('Get properties from LabelEncoder', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const classes = scaler.classes

    assert.deepEqual(classes, [1, 2, 'boy', 'git'])
  })
})
