import { dfd } from '../../globals'

/**
 * Loads the Boston housing dataset (regression). Samples 506, features 13.
 * @example
 * ```typescript
    import { loadBoston } from 'scikitjs'

    let df = await loadBoston()
    df.print()
    ```
 */
export async function loadBoston(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/boston.csv', {
    download: true
  })
}

/**
 * Loads the Iris dataset (classification).
 * This is a very easy multi-class classification dataset. Samples 150, Classes 3, Features 4.
 * @example
 * ```typescript
    import { loadIris } from 'scikitjs'

    let df = await loadIris()
    df.print()
    ```
 */
export async function loadIris(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/iris.csv', {
    download: true
  })
}

/**
 * Loads the Wine dataset (classification).
 * This is a very easy multi-class classification dataset. Samples 178, Classes 3, Features 13.
 * @example
 * ```typescript
    import { loadWine } from 'scikitjs'

    let df = await loadWine()
    df.print()
    ```
 */

export async function loadWine(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/wine.csv', {
    download: true
  })
}

/**
 * Loads the Diabetes dataset (regression).
 * Samples 442, Features 10.
 * @example
 * ```typescript
    import { loadDiabetes } from 'scikitjs'

    let df = await loadDiabetes()
    df.print()
    ```
 */

export async function loadDiabetes(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/diabetes.csv', {
    download: true
  })
}

/**
 * Loads the Breast Cancer Wisconsin dataset (classification).
 * Samples 569, Features 30.
 * @example
 * ```typescript
    import { loadBreastCancer } from 'scikitjs'

    let df = await loadBreastCancer()
    df.print()
    ```
 */

export async function loadBreastCancer(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/breast_cancer.csv', {
    download: true
  })
}

/**
 * Loads the Digit dataset (classification).
 * Samples 1797, Features 64. Each sample is an 8x8 image
 * @example
 * ```typescript
    import { loadDigits } from 'scikitjs'

    let df = await loadDigits()
    df.print()
    ```
 */
export async function loadDigits(): Promise<dfd.DataFrame> {
  return await dfd.read_csv('http://scikitjs.org/data/digits.csv', {
    download: true
  })
}

/**
 * Loads the California housing dataset (regression).
 *
 * Samples 20640, Features 8.
 */

export async function fetchCaliforniaHousing(): Promise<dfd.DataFrame> {
  return await dfd.read_csv(
    'http://scikitjs.org/data/california_housing.csv',
    {
      download: true
    }
  )
}
