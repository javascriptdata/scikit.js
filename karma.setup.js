// the jest.fn() API
let jest = require('jest-mock')
// The matchers API
let expect = require('expect')

// Add missing Jest functions
window.test = window.it
window.test.each = (inputs) => (testName, test) =>
  inputs.forEach((args) => window.it(testName, () => test(...args)))
window.test.todo = function () {
  return undefined
}
window.jest = jest
window.expect = expect
