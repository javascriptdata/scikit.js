// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github')
const darkCodeTheme = require('prism-react-renderer/themes/dracula')

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Scikit.js',
  tagline: 'Machine Learning for Javascript',
  url: 'https://scikitjs.org',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'opensource9ja', // Usually your GitHub org/user name.
  projectName: 'scikit.js', // Usually your repo name.

  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/opensource9ja/scikit.js/docs'
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl: 'https://github.com/opensource9ja/scikit.js/docs/blog/'
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css')
        }
      })
    ]
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: '',
        logo: {
          alt: 'Scikit.js Logo',
          src: 'img/sciKitLogo.svg'
        },
        items: [
          { to: '/', label: 'Home', position: 'right', exact: true },
          {
            type: 'doc',
            docId: 'intro',
            position: 'right',
            label: 'Tutorial'
          },
          {
            href: 'https://github.com/opensource9ja/scikit.js',
            label: 'GitHub',
            position: 'right'
          }
        ]
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro'
              }
            ]
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/scikitjs'
              }
            ]
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/opensource9ja/scikit.js'
              }
            ]
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Scikit.js`
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme
      }
    }),
  plugins: [
    [
      'docusaurus-plugin-typedoc',

      // Plugin / TypeDoc options
      {
        entryPoints: ['../scikitjs-node/src/index.ts'],
        tsconfig: '../scikitjs-node/tsconfig.json'
      }
    ]
  ]
}

module.exports = config
