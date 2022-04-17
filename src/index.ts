/**
*  @license
* Copyright 2021, JsData. All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==========================================================================
*/
export { KNeighborsRegressor } from './neighbors/kNeighborsRegressor'
export {
  LinearRegression,
  LinearRegressionParams
} from './linear_model/linearRegression'
export { LassoRegression, LassoParams } from './linear_model/lassoRegression'
export {
  RidgeRegression,
  RidgeRegressionParams
} from './linear_model/ridgeRegression'
export { ElasticNet, ElasticNetParams } from './linear_model/elasticNet'
export {
  LogisticRegression,
  LogisticRegressionParams
} from './linear_model/logisticRegression'
export * as metrics from './metrics/metrics'
export { DummyRegressor, DummyRegressorParams } from './dummy/dummyRegressor'
export {
  DummyClassifier,
  DummyClassifierParams
} from './dummy/dummyClassifier'
export { MinMaxScaler, MinMaxScalerParams } from './preprocessing/minMaxScaler'
export {
  StandardScaler,
  StandardScalerParams
} from './preprocessing/standardScaler'
export { MaxAbsScaler } from './preprocessing/maxAbsScaler'
export { SimpleImputer, SimpleImputerParams } from './impute/simpleImputer'
export {
  OneHotEncoder,
  OneHotEncoderParams
} from './preprocessing/oneHotEncoder'
export { LabelEncoder } from './preprocessing/labelEncoder'
export {
  OrdinalEncoder,
  OrdinalEncoderParams
} from './preprocessing/ordinalEncoder'
export { Normalizer, NormalizerParams } from './preprocessing/normalizer'
export { Pipeline, PipelineParams, makePipeline } from './pipeline/pipeline'
export {
  ColumnTransformer,
  ColumnTransformerParams
} from './compose/columnTransformer'
export { RobustScaler, RobustScalerParams } from './preprocessing/robustScaler'
export { KMeans, KMeansParams } from './cluster/kmeans'
export { Scikit1D, Scikit2D, ScikitVecOrMatrix } from './types'
export * as datasets from './datasets/datasets'
export {
  makeVotingRegressor,
  VotingRegressor,
  VotingRegressorParams
} from './ensemble/votingRegressor'
export {
  makeVotingClassifier,
  VotingClassifier,
  VotingClassifierParams
} from './ensemble/votingClassifier'
export { LinearSVC, LinearSVCParams } from './svm/linearSVC'
export { LinearSVR, LinearSVRParams } from './svm/linearSVR'

// Comment these out until our libsvm version doesn't ship with fs / path subdependencies
// They were stopping the browser build from being built
// export { SVR, SVRParams } from './svm/SVR'
// export { SVC, SVCParams } from './svm/SVC'
export { GaussianNB } from './naive_bayes/gaussianNaiveBayes'
export {
  DecisionTreeClassifier,
  DecisionTreeClassifierParams,
  DecisionTreeRegressor,
  DecisionTreeRegressorParams
} from './tree/decisiontree'
