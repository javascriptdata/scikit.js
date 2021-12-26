import { dfd } from '../../globals'
import * as https from 'https'

const load = (url: string) => {
  return dfd.read_csv(url, { download: true })

//  return new Promise<dfd.DataFrame>((resolve, fail) => {
//    const request = https.get(url, (response) => {
//      const code = response.statusCode as number
//      if (!(200 <= code && code < 300)) {
//        fail( new Error(`datasets.load: Sever responded with ${code}.`) )
//      }
//      else {
//        resolve( dfd.read_csv(response as any, {}) )
//      }
//    })
//    request.on('error', (err) => fail(err))
//  })
}

export const loadBoston = () => load('https://scikitjs.org/data/boston.csv')

export const loadIris = () => load('https://scikitjs.org/data/iris.csv')

export const loadWine = () => load('https://scikitjs.org/data/wine.csv')

export const loadDiabetes = () => load('https://scikitjs.org/data/diabetes.csv')

export const loadBreastCancer = () => load('https://scikitjs.org/data/breast_cancer.csv')

export const loadDigits = () => load('https://scikitjs.org/data/digits.csv')

export const fetchCaliforniaHousing = () => load('https://scikitjs.org/data/california_housing.csv')
