import { Scikit1D, Scikit2D } from '../types'

export interface KNeighborsRegressorParams {
  /** Number of neighbors that we use to decide this point. **default = 5 **
   */
  nNeighbors?: number

  /** Strategy for deciding how to weight points **default = undefined**
   */
  weights?: 'uniform'

  /** Algorithm used to compute nearest neighbors **default = brute** */
  algorithm?: 'brute'
}

export class KNeighborsRegressor {
  constructor({
    nNeighbors = 5,
    weights = 'uniform',
    algorithm = 'brute'
  }: KNeighborsRegressorParams = {}) {
    // does stuff
  }
  fit(X: Scikit2D, y: Scikit1D) {
    // does stuff
  }
  predict(X: Scikit2D) {
    // does stuff
  }
}
