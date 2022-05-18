import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  setBackend,
  fromJSON
} from '../index'
import { dataUrls } from '../datasets/datasets'
import * as dfd from 'danfojs-node'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('DecisionTree', function () {
  it('Use the DecisionTree (toy)', async function () {
    let X = [
      [-2, -1],
      [-1, -1],
      [-1, -2],
      [1, 1],
      [1, 2],
      [2, 1]
    ]
    let y = [0, 0, 0, 1, 1, 1]
    let T = [
      [-1, -1],
      [2, 2],
      [3, 2]
    ]
    let true_result = [0, 1, 1]
    let tree_classifier = new DecisionTreeClassifier()
    tree_classifier.fit(X, y)
    expect(tree_classifier.predict(T)).toEqual(true_result)
  }, 1000)
  it('Use the DecisionTree (toy 2)', async function () {
    let X = [
      [-2, -1],
      [-1, -1],
      [-1, -2],
      [1, 1],
      [1, 2],
      [2, 1]
    ]
    let y = [-1, -1, -1, 1, 1, 1]
    let T = [
      [-1, -1],
      [2, 2],
      [3, 2]
    ]
    let true_result = [-1, 1, 1]
    let tree_classifier = new DecisionTreeClassifier()
    tree_classifier.fit(X, y)
    expect(tree_classifier.predict(T)).toEqual(true_result)
  }, 1000)

  it('Use the DecisionTree (xor)', async function () {
    let X = []
    let y = []
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        X.push([i, j])
      }
    }
    for (let i = 0; i < 100; i++) {
      y.push(i < 5 ? 1 : 0)
    }

    let tree_classifier = new DecisionTreeClassifier()
    tree_classifier.fit(X, y)

    expect(tree_classifier.tree.nodes.length).toEqual(5)
    expect(tree_classifier.score(X, y)).toEqual(1.0)
  }, 1000)
  it('Deals with bad y input', async function () {
    let X = [
      [2, -1],
      [-1, -1],
      [-1, -2]
    ]
    let y1 = [0, 0, '0']
    let y2 = [0, {}, 0]
    let y3 = [0, 0, 4.2]
    let y4 = [NaN, 0, null]
    let y5 = [0, 2, 3]
    let y6: number[] = []

    let tree_classifier = new DecisionTreeClassifier()
    expect(() => tree_classifier.fit(X as any, y1 as any)).toThrow()
    expect(() => tree_classifier.fit(X as any, y2 as any)).toThrow()
    expect(() => tree_classifier.fit(X as any, y3 as any)).toThrow()
    expect(() => tree_classifier.fit(X as any, y4 as any)).toThrow()
    expect(() => tree_classifier.fit(X as any, y5 as any)).not.toThrow()
    expect(() => tree_classifier.fit(X as any, y6)).toThrow()
  }, 1000)
  it('Deals with bad X input', async function () {
    let X1 = [
      [2, -1],
      [-1, '-1']
    ]
    let X2 = [
      [2, Number.POSITIVE_INFINITY],
      [-1, 0]
    ]
    let X3 = [
      [2, -1],
      [-1, NaN]
    ]
    let y1 = [0, 0]

    let tree_classifier = new DecisionTreeClassifier()
    expect(() => tree_classifier.fit(X1 as any, y1 as any)).toThrow()
    expect(() => tree_classifier.fit(X2 as any, y1 as any)).toThrow()
    expect(() => tree_classifier.fit(X3 as any, y1 as any)).toThrow()
  }, 1000)
  it('Use the DecisionTreeRegressor', async function () {
    let X = [[1], [2], [3], [4], [5], [6], [7], [8]]
    let y = [-1, -1, -1, -1, 2, 2, 2, 2]

    let tree_regressor = new DecisionTreeRegressor()
    tree_regressor.fit(X, y)

    expect(tree_regressor.tree.nodes.length).toEqual(3)
    expect(tree_regressor.predict([[3]])).toEqual([-1])
    expect(tree_regressor.score(X, y)).toEqual(1.0)
  }, 1000)
  it('Medium sized Iris example', async function () {
    /*
    [[[cog
    import cog
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    data = load_iris()
    X, y = data.data, data.target

    results = []
    for criterion in ['entropy', 'gini']:
      for max_depth in [1, 2, 3, 4, None]:
        for min_samples_leaf in [1, 2, 4, 8]:
          clf = DecisionTreeClassifier(criterion = criterion,
                                      max_depth = max_depth,
                                      min_samples_leaf = min_samples_leaf)
          clf.fit(X, y)
          results.append({
            "criterion": criterion,
            "max_depth": "undefined" if max_depth is None else max_depth,
            "min_samples_leaf": min_samples_leaf,
            "leaf_count": clf.get_n_leaves(),
            "score": clf.score(X, y)
          })

    cog.outl('let results = ' + str(results))
      ]]]*/
    let results = [
      {
        criterion: 'entropy',
        max_depth: 1,
        min_samples_leaf: 1,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'entropy',
        max_depth: 1,
        min_samples_leaf: 2,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'entropy',
        max_depth: 1,
        min_samples_leaf: 4,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'entropy',
        max_depth: 1,
        min_samples_leaf: 8,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'entropy',
        max_depth: 2,
        min_samples_leaf: 1,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: 2,
        min_samples_leaf: 2,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: 2,
        min_samples_leaf: 4,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: 2,
        min_samples_leaf: 8,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: 3,
        min_samples_leaf: 1,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'entropy',
        max_depth: 3,
        min_samples_leaf: 2,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'entropy',
        max_depth: 3,
        min_samples_leaf: 4,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'entropy',
        max_depth: 3,
        min_samples_leaf: 8,
        leaf_count: 5,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: 4,
        min_samples_leaf: 1,
        leaf_count: 8,
        score: 0.9933333333333333
      },
      {
        criterion: 'entropy',
        max_depth: 4,
        min_samples_leaf: 2,
        leaf_count: 7,
        score: 0.98
      },
      {
        criterion: 'entropy',
        max_depth: 4,
        min_samples_leaf: 4,
        leaf_count: 6,
        score: 0.9733333333333334
      },
      {
        criterion: 'entropy',
        max_depth: 4,
        min_samples_leaf: 8,
        leaf_count: 6,
        score: 0.96
      },
      {
        criterion: 'entropy',
        max_depth: undefined,
        min_samples_leaf: 1,
        leaf_count: 9,
        score: 1.0
      },
      {
        criterion: 'entropy',
        max_depth: undefined,
        min_samples_leaf: 2,
        leaf_count: 7,
        score: 0.98
      },
      {
        criterion: 'entropy',
        max_depth: undefined,
        min_samples_leaf: 4,
        leaf_count: 6,
        score: 0.9733333333333334
      },
      {
        criterion: 'entropy',
        max_depth: undefined,
        min_samples_leaf: 8,
        leaf_count: 6,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 1,
        min_samples_leaf: 1,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'gini',
        max_depth: 1,
        min_samples_leaf: 2,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'gini',
        max_depth: 1,
        min_samples_leaf: 4,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'gini',
        max_depth: 1,
        min_samples_leaf: 8,
        leaf_count: 2,
        score: 0.6666666666666666
      },
      {
        criterion: 'gini',
        max_depth: 2,
        min_samples_leaf: 1,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 2,
        min_samples_leaf: 2,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 2,
        min_samples_leaf: 4,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 2,
        min_samples_leaf: 8,
        leaf_count: 3,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 3,
        min_samples_leaf: 1,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'gini',
        max_depth: 3,
        min_samples_leaf: 2,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'gini',
        max_depth: 3,
        min_samples_leaf: 4,
        leaf_count: 5,
        score: 0.9733333333333334
      },
      {
        criterion: 'gini',
        max_depth: 3,
        min_samples_leaf: 8,
        leaf_count: 5,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: 4,
        min_samples_leaf: 1,
        leaf_count: 8,
        score: 0.9933333333333333
      },
      {
        criterion: 'gini',
        max_depth: 4,
        min_samples_leaf: 2,
        leaf_count: 7,
        score: 0.98
      },
      {
        criterion: 'gini',
        max_depth: 4,
        min_samples_leaf: 4,
        leaf_count: 6,
        score: 0.9733333333333334
      },
      {
        criterion: 'gini',
        max_depth: 4,
        min_samples_leaf: 8,
        leaf_count: 6,
        score: 0.96
      },
      {
        criterion: 'gini',
        max_depth: undefined,
        min_samples_leaf: 1,
        leaf_count: 9,
        score: 1.0
      },
      {
        criterion: 'gini',
        max_depth: undefined,
        min_samples_leaf: 2,
        leaf_count: 7,
        score: 0.98
      },
      {
        criterion: 'gini',
        max_depth: undefined,
        min_samples_leaf: 4,
        leaf_count: 6,
        score: 0.9733333333333334
      },
      {
        criterion: 'gini',
        max_depth: undefined,
        min_samples_leaf: 8,
        leaf_count: 6,
        score: 0.96
      }
    ]
    /*[[[end]]]*/
    let df = await dfd.readCSV(dataUrls.loadIris)
    let y = df['target'].values
    let X = df.drop({ columns: 'target' }).values

    results.forEach((el) => {
      let clf = new DecisionTreeClassifier({
        criterion: el.criterion as any,
        maxDepth: el.max_depth,
        minSamplesLeaf: el.min_samples_leaf
      })
      clf.fit(X as number[][], y)
      expect(clf.getNLeaves()).toEqual(el.leaf_count)
      expect(clf.score(X as number[][], y)).toBeCloseTo(el.score)
    })
  })
  it('Medium sized Boston example', async function () {
    /*
    [[[cog
    import cog
    import sklearn
    from sklearn.datasets import load_boston
    from sklearn.tree import DecisionTreeRegressor
    data = load_boston()
    X, y = data.data, data.target
    results = []
    for max_depth in [1, 2, 3, 4, None]:
      for min_samples_leaf in [2, 4, 8, 16]:
        clf = DecisionTreeRegressor(max_depth = max_depth,
                                    min_samples_leaf = min_samples_leaf)
        clf.fit(X, y)
        results.append({
          "max_depth": "undefined" if max_depth is None else max_depth,
          "min_samples_leaf": min_samples_leaf,
          "leaf_count": clf.get_n_leaves(),
          "score": clf.score(X, y)
        })

    cog.outl('let results = ' + str(results))
      ]]]*/
    let results = [
      {
        max_depth: 1,
        min_samples_leaf: 2,
        leaf_count: 2,
        score: 0.4527442007436526
      },
      {
        max_depth: 1,
        min_samples_leaf: 4,
        leaf_count: 2,
        score: 0.4527442007436526
      },
      {
        max_depth: 1,
        min_samples_leaf: 8,
        leaf_count: 2,
        score: 0.4527442007436526
      },
      {
        max_depth: 1,
        min_samples_leaf: 16,
        leaf_count: 2,
        score: 0.4527442007436526
      },
      {
        max_depth: 2,
        min_samples_leaf: 2,
        leaf_count: 4,
        score: 0.695574477973027
      },
      {
        max_depth: 2,
        min_samples_leaf: 4,
        leaf_count: 4,
        score: 0.695574477973027
      },
      {
        max_depth: 2,
        min_samples_leaf: 8,
        leaf_count: 4,
        score: 0.6955744779730269
      },
      {
        max_depth: 2,
        min_samples_leaf: 16,
        leaf_count: 4,
        score: 0.6955744779730269
      },
      {
        max_depth: 3,
        min_samples_leaf: 2,
        leaf_count: 8,
        score: 0.8156207236848065
      },
      {
        max_depth: 3,
        min_samples_leaf: 4,
        leaf_count: 8,
        score: 0.8106934802243154
      },
      {
        max_depth: 3,
        min_samples_leaf: 8,
        leaf_count: 8,
        score: 0.7766125794987506
      },
      {
        max_depth: 3,
        min_samples_leaf: 16,
        leaf_count: 7,
        score: 0.7670579475328108
      },
      {
        max_depth: 4,
        min_samples_leaf: 2,
        leaf_count: 14,
        score: 0.8762193275488561
      },
      {
        max_depth: 4,
        min_samples_leaf: 4,
        leaf_count: 13,
        score: 0.8655911018334532
      },
      {
        max_depth: 4,
        min_samples_leaf: 8,
        leaf_count: 14,
        score: 0.8176127442290948
      },
      {
        max_depth: 4,
        min_samples_leaf: 16,
        leaf_count: 10,
        score: 0.7981355077279791
      },
      {
        max_depth: undefined,
        min_samples_leaf: 2,
        leaf_count: 217,
        score: 0.9795789095271915
      },
      {
        max_depth: undefined,
        min_samples_leaf: 4,
        leaf_count: 100,
        score: 0.9483838254117538
      },
      {
        max_depth: undefined,
        min_samples_leaf: 8,
        leaf_count: 48,
        score: 0.8786522413769035
      },
      {
        max_depth: undefined,
        min_samples_leaf: 16,
        leaf_count: 24,
        score: 0.8325748991177735
      }
    ]
    /*[[[end]]]*/
    let df = await dfd.readCSV(dataUrls.loadBoston)
    let y = df['target'].values
    let X = df.drop({ columns: 'target' }).values

    results.forEach((el) => {
      let clf = new DecisionTreeRegressor({
        criterion: 'squared_error',
        maxDepth: el.max_depth,
        minSamplesLeaf: el.min_samples_leaf
      })
      clf.fit(X as number[][], y)
      expect(clf.getNLeaves()).toEqual(el.leaf_count)
      expect(clf.score(X as number[][], y)).toBeCloseTo(el.score)
    })
  })
  it('Use the DecisionTree (toy)', async function () {
    let X = [
      [-2, -1],
      [-1, -1],
      [-1, -2],
      [1, 1],
      [1, 2],
      [2, 1]
    ]
    let y = [0, 0, 0, 1, 1, 1]
    let T = [
      [-1, -1],
      [2, 2],
      [3, 2]
    ]
    let true_result = [0, 1, 1]
    let tree_classifier = new DecisionTreeClassifier()
    tree_classifier.fit(X, y)

    const serial = await tree_classifier.toJSON()
    const newTree = await fromJSON(serial)
    expect(newTree.predict(T)).toEqual(true_result)
  }, 1000)
})
