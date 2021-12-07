import { dfd } from '../../globals'

export async function loadBoston() {
  return await dfd.read_csv('http://scikitjs.org/data/boston.csv', {})
}

export async function loadIris() {
  return await dfd.read_csv('http://scikitjs.org/data/iris.csv', {})
}

export async function loadWine() {
  return await dfd.read_csv('http://scikitjs.org/data/wine.csv', {})
}

export async function loadDiabetes() {
  return await dfd.read_csv('http://scikitjs.org/data/diabetes.csv', {})
}

export async function loadBreastCancer() {
  return await dfd.read_csv('http://scikitjs.org/data/breast_cancer.csv', {})
}

export async function loadDigits() {
  return await dfd.read_csv('http://scikitjs.org/data/digits.csv', {})
}

export async function fetchCaliforniaHousing() {
  return await dfd.read_csv(
    'http://scikitjs.org/data/california_housing.csv',
    {}
  )
}
