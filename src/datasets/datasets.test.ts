import { loadBoston } from './datasets'

describe('test', function () {
  it('loadBoston test', async function () {
    let Array2D = await loadBoston()
    expect(Array2D.length).toEqual(507)
    expect(Array2D[0].length).toEqual(14)
  })
})
