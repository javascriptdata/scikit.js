import { assert } from 'chai'
import { LabelEncoder } from '../../../dist'
import { dfd } from '../../globals'

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
    const nClasses = scaler.nClasses

    let obj = Array.from(classes).reduce(
      (obj, [key, value]) => Object.assign(obj, { [key as any]: value }), // Be careful! Maps can have non-String keys; object literals can't.
      {}
    )

    assert.deepEqual(obj, { 1: 0, 2: 1, boy: 2, git: 3 })
    assert.equal(nClasses, 4)
  })
})
