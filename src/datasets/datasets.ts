import { dfd } from '../shared/globals'

export async function loadBoston() {
  return await dfd.read_csv('http://scikitjs.org/data/boston.csv', {
    download: true
  })
}

export async function loadIris() {
  return await dfd.read_csv('http://scikitjs.org/data/iris.csv', {
    download: true
  })
}

export async function loadWine() {
  return await dfd.read_csv('http://scikitjs.org/data/wine.csv', {
    download: true
  })
}

export async function loadDiabetes() {
  return await dfd.read_csv('http://scikitjs.org/data/diabetes.csv', {
    download: true
  })
}

export async function loadBreastCancer() {
  return await dfd.read_csv('http://scikitjs.org/data/breast_cancer.csv', {
    download: true
  })
}

export async function loadDigits() {
  return await dfd.read_csv('http://scikitjs.org/data/digits.csv', {
    download: true
  })
}

export async function fetchCaliforniaHousing() {
  return await dfd.read_csv(
    'http://scikitjs.org/data/california_housing.csv',
    {
      download: true
    }
  )
}
