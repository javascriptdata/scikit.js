module.exports = function (config) {
  config.set({
    plugins: ['karma-webpack', 'karma-jasmine', 'karma-chrome-launcher'],

    // base path that will be used to resolve all patterns (eg. files, exclude)
    basePath: '',

    // frameworks to use
    // available frameworks: https://npmjs.org/browse/keyword/karma-adapter
    frameworks: ['jasmine'],

    // list of files / patterns to load in the browser
    // Here I'm including all of the the Jest tests which are all under the __tests__ directory.
    // You may need to tweak this pattern to find your test files/
    files: ['./karma-setup.js', 'out.js'],
    browsers: ['ChromeHeadless'],
    singleRun: true
  })
}
