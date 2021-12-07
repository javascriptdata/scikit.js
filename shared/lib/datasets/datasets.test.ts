import { assert } from 'chai'
import { describe, it } from 'mocha'
import {
  loadBoston,
  loadIris,
  loadWine,
  loadBreastCancer,
  loadDiabetes,
  loadDigits
} from './datasets'

describe('Test all the dataset functions', function () {
  this.timeout(5000)
  it('Can download the boston data', async function () {
    const boston = await loadBoston()
    assert.deepEqual(boston.columns, [
      'CRIM',
      'ZN',
      'INDUS',
      'CHAS',
      'NOX',
      'RM',
      'AGE',
      'DIS',
      'RAD',
      'TAX',
      'PTRATIO',
      'B',
      'LSTAT',
      'target'
    ])
    assert.deepEqual(boston.shape, [506, 14])
  })
  it('Can download the iris data', async function () {
    const iris = await loadIris()
    assert.deepEqual(iris.columns, [
      'sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)',
      'target'
    ])
    assert.deepEqual(iris.shape, [150, 5])
  })
  it('Can download the wine data', async function () {
    const wine = await loadWine()
    assert.deepEqual(wine.columns, [
      'alcohol',
      'malic_acid',
      'ash',
      'alcalinity_of_ash',
      'magnesium',
      'total_phenols',
      'flavanoids',
      'nonflavanoid_phenols',
      'proanthocyanins',
      'color_intensity',
      'hue',
      'od280/od315_of_diluted_wines',
      'proline',
      'target'
    ])
    assert.deepEqual(wine.shape, [178, 14])
  })
  it('Can download the breast_cancer data', async function () {
    const breast = await loadBreastCancer()
    assert.deepEqual(breast.columns, [
      'mean radius',
      'mean texture',
      'mean perimeter',
      'mean area',
      'mean smoothness',
      'mean compactness',
      'mean concavity',
      'mean concave points',
      'mean symmetry',
      'mean fractal dimension',
      'radius error',
      'texture error',
      'perimeter error',
      'area error',
      'smoothness error',
      'compactness error',
      'concavity error',
      'concave points error',
      'symmetry error',
      'fractal dimension error',
      'worst radius',
      'worst texture',
      'worst perimeter',
      'worst area',
      'worst smoothness',
      'worst compactness',
      'worst concavity',
      'worst concave points',
      'worst symmetry',
      'worst fractal dimension',
      'target'
    ])
    assert.deepEqual(breast.shape, [569, 31])
  })
  it('Can download the digits data', async function () {
    const digits = await loadDigits()
    let array = []
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        array.push(`pixel_${i}_${j}`)
      }
    }
    array.push('target')
    assert.deepEqual(digits.columns, array)
    assert.deepEqual(digits.shape, [1797, 65])
  })
  it('Can download the diabetes data', async function () {
    const diabetes = await loadDiabetes()
    assert.deepEqual(diabetes.columns, [
      'age',
      'sex',
      'bmi',
      'bp',
      's1',
      's2',
      's3',
      's4',
      's5',
      's6',
      'target'
    ])
    assert.deepEqual(diabetes.shape, [442, 11])
  })
})
