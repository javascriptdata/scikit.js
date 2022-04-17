import { LabelEncoder } from './labelEncoder'
import * as dfd from 'danfojs-node'

describe('LabelEncoder', function () {
  it('LabelEncoder works for Series', function () {
    const sf = new dfd.Series([1, 2, 2, 6])
    const scaler = new LabelEncoder()
    scaler.fit(sf)
    const expected = [0, 1, 1, 2]
    expect(scaler.transform(sf).arraySync()).toEqual(expected)
  })
  it('LabelEncoder works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const expected = [0, 1, 1, 2, 3, 3]
    expect(scaler.transform(sf as any).arraySync()).toEqual(expected)
  })
  it('fitTransform works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    const result = scaler.fitTransform(sf as any)
    const expected = [0, 1, 1, 2, 3, 3]
    expect(result.arraySync()).toEqual(expected)
  })
  it('inverseTransform works for 1D array', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const result = scaler.inverseTransform([0, 1, 1, 2, 3, 3])
    expect(result).toEqual([1, 2, 2, 'boy', 'git', 'git'])
  })
  it('Get properties from LabelEncoder', function () {
    const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf as any)
    const classes = scaler.classes

    expect(classes).toEqual([1, 2, 'boy', 'git'])
  })
})
