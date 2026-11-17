// 基础配置
export const BASE_CONFIG = {
  title: "Trevor's Blog",
  subtitle: "Trevor's Blog",
  description: '这里记录生活、学习，以及值得留下的想法。',
  author: 'Trevor',
  favicon: '/favicon.png',
  avatar: '/avatar.png',
  site_url: 'https://trevoewu.github.io',
  create_time: '2023-04-21',
}

// 配色方案
export type PaletteName = 'paper' | 'mono' | 'normal'

export const THEME_CONFIG = {
  palette: 'paper',
} as const satisfies { palette: PaletteName }

// 文章页配置
export const POST_CONFIG = {
  toc: true,
}

// 导航栏配置
export const HEADER_CONFIG = {
  nav: [
    { label: '归档', href: '/archive' },
    { label: '分类', href: '/category' },
    { label: '友链', href: '/link' },
    { label: '关于', href: '/about' },
  ],
}

// 页脚配置
export const FOOTER_CONFIG = {
  link: [
    { name: 'GitHub', url: 'https://github.com/Trevoewu' },
    { name: 'RSS', url: '/rss.xml' },
    { name: 'Astro', url: 'https://astro.build/' },
  ]
}

// 评论配置
export const COMMENT_CONFIG = {
  enable: false,
  provider: 'waline',
  waline: {
    serverURL: 'https://example.cn/',
  },
}
