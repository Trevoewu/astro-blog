export const SITE_TITLE = "Trevor's Blog";
export const SITE_DESCRIPTION = 'Notes on life, learning, and ideas worth keeping.';
export const SITE_URL = process.env.SITE_URL || 'https://trevor.ccwu.cc';

export const PORTRAIT: { light: string; dark?: string } | undefined = {
  light: '/portrait-light.webp',
  dark: '/portrait-dark.webp',
};

export const LINKS: { label: string; labelZh?: string; href: string }[] = [
  { label: 'GitHub', href: 'https://github.com/Trevoewu' },
  { label: 'Links', labelZh: '友链', href: '/links/' },
];
