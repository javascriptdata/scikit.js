import typescript from '@rollup/plugin-typescript'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import builtins from 'rollup-plugin-node-builtins'
import globals from 'rollup-plugin-node-globals'
import replace from '@rollup/plugin-replace'
import terser from 'rollup-plugin-terser-js'
import json from '@rollup/plugin-json'
import pkg from './package.json'
import alias from '@rollup/plugin-alias'

const name = 'scikitjs'

function getOutput({ minify = false }) {
  const output = [
    {
      file: pkg.output.web,
      format: 'iife',
      exports: 'named',
      name,
      sourcemap: true
    }
  ]
  return output.map((item) => {
    const itemFileArray = item.file.split('.')
    if (minify) {
      itemFileArray.splice(itemFileArray.length - 1, 0, 'min')
    }
    item.file = itemFileArray.join('.')
    item.sourcemap = false
    return item
  })
}

function getPlugins({ minify = false }) {
  const plugins = []

  plugins.push(
    ...[
      alias({
        resolve: ['.js', '.ts'],
        entries: {
          '@tensorflow/tfjs-node': '@tensorflow/tfjs',
          'danfojs-node': 'danfojs'
        }
      })
    ]
  )

  plugins.push(
    ...[
      json(),
      replace({
        preventAssignment: true,
        delimiters: ['', '']
      }),
      replace({
        preventAssignment: true,
        'process.env.NODE_ENV': minify
          ? JSON.stringify('production')
          : JSON.stringify('development')
      }),
      builtins({}),
      resolve({
        preferBuiltins: true
      }),
      typescript({
        noEmitOnError: false,
        declaration: false,
        declarationDir: null,
        allowJs: true,
        downlevelIteration: true,
        target: 'es5', //legacy ? "es5" : "esnext",
        tsconfig: './tsconfig.build.json'
      }),
      commonjs({
        extensions: ['.js']
      }), // so Rollup can convert `ms` to an ES module
      globals({})
    ]
  )
  if (minify) {
    plugins.push(
      terser({
        sourcemaps: true,
        compress: true,
        mangle: true,
        verbose: true
      })
    )
    return plugins
  }
  return plugins
}

export default [
  //web
  {
    // input: "dist/umd/index.js",
    input: 'src/index.ts',
    output: getOutput({
      minify: false
    }),
    plugins: getPlugins({
      minify: false
    })
  },
  //web minified
  {
    input: 'src/index.ts',
    output: getOutput({
      minify: true
    }),
    plugins: getPlugins({
      minify: true
    })
  }
]
