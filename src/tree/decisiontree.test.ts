import { DecisionTreeClassifier, DecisionTreeRegressor } from './decisiontree'

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
  }, 80000)

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

    expect(tree_classifier.tree_.nodes.length).toEqual(5)
    expect(tree_classifier.score(X, y)).toEqual(1.0)
  }, 1000)
  // it('Use the DecisionTreeRegressor', async function () {
  //   let X = [[1], [2], [3], [4], [5], [6], [7], [8]]
  //   let y = [1, 1, 1, 1, 2, 2, 2, 2]

  //   let tree_classifier = new DecisionTreeRegressor()
  //   tree_classifier.fit(X, y)

  //   expect(tree_classifier.tree_.nodes.length).toEqual(1)
  //   expect(tree_classifier.score(X, y)).toEqual(1.0)
  // }, 1000)
})
