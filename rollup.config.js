import typescript from '@rollup/plugin-typescript'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import builtins from 'rollup-plugin-node-builtins'
import globals from 'rollup-plugin-node-globals'
import replace from '@rollup/plugin-replace'
import terser from 'rollup-plugin-terser-js'
// import sucrase from '@rollup/plugin-sucrase';
import json from '@rollup/plugin-json'
import pkg from "./package.json"
import alias from '@rollup/plugin-alias'

const name = 'scikit'
const external = [
  // "react-dom",
]
const serverExternal = [
  "@tensorflow/tfjs",
  "@tensorflow/tfjs-node",
  "danfojs",
  "lodash",
  "mathjs",
  "seedrandom",
  "simple-statistics"
]
const windowGlobals = {
  // react: "React",
}

function getOutput({ minify = false, server = false, legacy = false }) {
  const output = server ?
    [ {
      file: pkg.output.node,
      format: "cjs",
      exports: "named",
      name,
      sourcemap: true
    },
    {
      file: pkg.main,
      format: "es",
      exports: "named",
      name,
      sourcemap: true
    }
    ] :
    [ {
      file: pkg.output.web,
      format: "umd",
      exports: "named",
      name,
      globals:windowGlobals,
      sourcemap: true
    },
    {
      file: pkg.output.web,
      format: "iife",
      exports: "named",
      name,
      globals:windowGlobals,
      sourcemap: true
    }
  ]
  return output.map((item) => {
    // console.log({item})
    const itemFileArray = item.file.split('.')
    if (minify) {
      itemFileArray.splice(itemFileArray.length - 1, 0, legacy ? 'legacy-min' : 'min')
    } else if (legacy){
      itemFileArray.splice(itemFileArray.length - 1, 0, legacy ? 'legacy' : '')
    }
    item.file = itemFileArray.join('.')
    item.sourcemap = false
    return item
  })
  // return output;
}

function getPlugins({
  minify = false,
  browser = false,
  // server = false,
  legacy = false
}) {
  const plugins = [ ]
  if (browser) {
    plugins.push(
      ...[
        alias({
          resolve: ['.js', '.ts'],
          entries: {
            '@tensorflow/tfjs-node': '@tensorflow/tfjs'
            // 'natural':'./node_modules/@jsonstack/data/src/stub.ts',
            // 'probability-distributions': './src/stubs/prob_stub-temp.js',
            // 'async_hooks': './node_modules/@jsonstack/data/src/async_hook_stub.ts',
            // 'tsne-js': 'tsne-js/build/tsne.min.js'
          }
        })
      ])
  }
  plugins.push(...[
    json(),
    replace({
      preventAssignment: true,
      delimiters: ['', '']
    }),
    replace({
      preventAssignment: true,
      'process.env.NODE_ENV': minify ?
        JSON.stringify('production') : JSON.stringify('development')
    }),
    builtins({}),
    // // nodePolyfills({}),
    resolve({
      preferBuiltins: true
    }),
    typescript({
      noEmitOnError: false,
      declaration: false,
      declarationDir: null,
      allowJs:true,
      target: legacy ? "es5" : "esnext"
    }),
    commonjs({
      extensions: ['.js']
    }), // so Rollup can convert `ms` to an ES module
    globals({
      // react: 'React',
    })
  ])
  if (minify) {
    const minifyPlugins = [

    ].concat(plugins,
      [
        terser({
          sourcemaps: true,
          compress: true,
          mangle: true,
          verbose: true
        })
      ])
    return minifyPlugins
  }
  return plugins
}


export default [
  //web
  {
    input: "src/lib/index.ts",
    output: getOutput({
      minify: false,
      server: false
    }),
    external,
    plugins: getPlugins({
      minify: false,
      browser:true
    })
  },
  //web minified
  {
    input: "src/lib/index.ts",
    output: getOutput({
      minify: true,
      server: false
    }),
    external,
    plugins: getPlugins({
      minify: true,
      browser:true
    })
  },
  //web - LEGACY
  {
    input: "src/lib/index.ts",
    output: getOutput({
      minify: false,
      server: false,
      legacy: true
    }),
    external,
    plugins: getPlugins({
      minify: false,
      browser: true,
      legacy: true
    })
  },
  //web minified - LEGACY
  {
    input: "src/lib/index.ts",
    output: getOutput({
      minify: true,
      server: false,
      legacy: true
    }),
    external,
    plugins: getPlugins({
      minify: true,
      browser: true,
      legacy: true
    })
  },
  //node cjs + esm
  {
    input: "src/lib/index.ts",
    output: getOutput({
      minify: false,
      server: true
    }),
    external:serverExternal,
    plugins: getPlugins({
      minify: false,
      server: true
    })
  }
]
