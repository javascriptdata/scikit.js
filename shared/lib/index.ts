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
import {
  LinearRegression,
  LinearRegressionParams
} from './estimators/linearRegression'
import { LassoRegression, LassoParams } from './estimators/lassoRegression'
import {
  RidgeRegression,
  RidgeRegressionParams
} from './estimators/ridgeRegression'
import { ElasticNet, ElasticNetParams } from './estimators/elasticNet'
import {
  LogisticRegression,
  LogisticRegressionParams
} from './estimators/logisticRegression'
import * as metrics from './metrics/metrics'
import { DummyRegressor, DummyRegressorParams } from './dummy/dummyRegressor'
import {
  DummyClassifier,
  DummyClassifierParams
} from './dummy/dummyClassifier'
import { MinMaxScaler, MinMaxScalerParams } from './preprocessing/minMaxScaler'
import {
  StandardScaler,
  StandardScalerParams
} from './preprocessing/standardScaler'
import { MaxAbsScaler } from './preprocessing/maxAbsScaler'
import { SimpleImputer, SimpleImputerParams } from './impute/simpleImputer'
import {
  OneHotEncoder,
  OneHotEncoderParams
} from './preprocessing/oneHotEncoder'
import { LabelEncoder } from './preprocessing/labelEncoder'
import {
  OrdinalEncoder,
  OrdinalEncoderParams
} from './preprocessing/ordinalEncoder'
import { Normalizer, NormalizerParams } from './preprocessing/normalizer'
import { Pipeline, PipelineParams, makePipeline } from './pipeline/pipeline'
import {
  ColumnTransformer,
  ColumnTransformerParams
} from './compose/columnTransformer'
import { RobustScaler, RobustScalerParams } from './preprocessing/robustScaler'
import { KMeans, KMeansParams } from './cluster/kmeans'
import { Scikit1D, Scikit2D, ScikitVecOrMatrix } from './types'
import {
  loadBoston,
  loadIris,
  loadWine,
  loadDiabetes,
  loadBreastCancer,
  loadDigits,
  fetchCaliforniaHousing
} from './datasets/datasets'
import {
  VotingRegressor,
  VotingRegressorParams
} from './ensemble/votingRegressor'

export {
  MinMaxScaler,
  MinMaxScalerParams,
  StandardScaler,
  StandardScalerParams,
  MaxAbsScaler,
  RobustScaler,
  RobustScalerParams,
  Normalizer,
  NormalizerParams,
  OneHotEncoder,
  OneHotEncoderParams,
  LabelEncoder,
  SimpleImputer,
  SimpleImputerParams,
  DummyRegressor,
  DummyRegressorParams,
  DummyClassifier,
  DummyClassifierParams,
  LinearRegression,
  LinearRegressionParams,
  LassoRegression,
  LassoParams,
  RidgeRegression,
  RidgeRegressionParams,
  LogisticRegression,
  LogisticRegressionParams,
  ElasticNet,
  ElasticNetParams,
  metrics,
  Pipeline,
  PipelineParams,
  makePipeline,
  ColumnTransformer,
  ColumnTransformerParams,
  OrdinalEncoder,
  OrdinalEncoderParams,
  KMeans,
  KMeansParams,
  VotingRegressor,
  VotingRegressorParams,
  loadBoston,
  loadDiabetes,
  loadIris,
  loadWine,
  loadBreastCancer,
  loadDigits,
  fetchCaliforniaHousing,
  Scikit1D,
  Scikit2D,
  ScikitVecOrMatrix
}
