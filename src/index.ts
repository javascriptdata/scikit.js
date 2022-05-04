import Serialize from './serialize'

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
export { KNeighborsRegressor } from './neighbors/KNeighborsRegressor'
export {
  LinearRegression,
  LinearRegressionParams
} from './linear_model/LinearRegression'
export { LassoRegression, LassoParams } from './linear_model/LassoRegression'
export {
  RidgeRegression,
  RidgeRegressionParams
} from './linear_model/RidgeRegression'
export { ElasticNet, ElasticNetParams } from './linear_model/ElasticNet'
export {
  LogisticRegression,
  LogisticRegressionParams
} from './linear_model/LogisticRegression'
export * as metrics from './metrics/metrics'
export { DummyRegressor, DummyRegressorParams } from './dummy/DummyRegressor'
export {
  DummyClassifier,
  DummyClassifierParams
} from './dummy/DummyClassifier'
export { MinMaxScaler, MinMaxScalerParams } from './preprocessing/MinMaxScaler'
export {
  StandardScaler,
  StandardScalerParams
} from './preprocessing/StandardScaler'
export { MaxAbsScaler } from './preprocessing/MaxAbsScaler'
export { SimpleImputer, SimpleImputerParams } from './impute/SimpleImputer'
export {
  OneHotEncoder,
  OneHotEncoderParams
} from './preprocessing/OneHotEncoder'
export { LabelEncoder } from './preprocessing/LabelEncoder'
export {
  OrdinalEncoder,
  OrdinalEncoderParams
} from './preprocessing/OrdinalEncoder'
export { Normalizer, NormalizerParams } from './preprocessing/Normalizer'
export { Pipeline, PipelineParams, makePipeline } from './pipeline/Pipeline'
export {
  ColumnTransformer,
  ColumnTransformerParams
} from './compose/ColumnTransformer'
export { RobustScaler, RobustScalerParams } from './preprocessing/RobustScaler'
export { KMeans, KMeansParams } from './cluster/KMeans'
export { Scikit1D, Scikit2D, ScikitVecOrMatrix } from './types'
export { dataUrls } from './datasets/datasets'
export {
  makeVotingRegressor,
  VotingRegressor,
  VotingRegressorParams
} from './ensemble/VotingRegressor'
export {
  makeVotingClassifier,
  VotingClassifier,
  VotingClassifierParams
} from './ensemble/VotingClassifier'
export { LinearSVC, LinearSVCParams } from './svm/LinearSVC'
export { LinearSVR, LinearSVRParams } from './svm/LinearSVR'

// Comment these out until our libsvm version doesn't ship with fs / path subdependencies
// They were stopping the browser build from being built
// export { SVR, SVRParams } from './svm/SVR'
// export { SVC, SVCParams } from './svm/SVC'
export { GaussianNB } from './naive_bayes/GaussianNB'
export {
  DecisionTreeClassifier,
  DecisionTreeClassifierParams,
  DecisionTreeRegressor,
  DecisionTreeRegressorParams
} from './tree/DecisionTree'

export { fromObject, Serialize } from './simpleSerializer'

export { ClassificationCriterion, RegressionCriterion } from './tree/Criterion'
export { Splitter } from './tree/Splitter'
export { DecisionTreeBase, DecisionTree } from './tree/DecisionTree'
