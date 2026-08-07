# Security

## Reporting

If you find a vulnerability, please use GitHub's private vulnerability
reporting on this repo (Security → Report a vulnerability) rather than opening
a public issue. Reports get a response within a week.

## Posture

NNvisual is a static, client-side app with a deliberately small attack surface:

- **Zero runtime dependencies.** Nothing is fetched from a CDN and no
  third-party code runs in the page. The only dev dependency is ESLint, updated
  weekly by Dependabot.
- **Strict CSP.** The page ships a Content-Security-Policy meta tag that allows
  same-origin scripts/styles/fonts only, with `object-src 'none'` and
  `base-uri 'none'`. When served on j4den.com, the site's headers additionally
  apply `X-Frame-Options: DENY`, `nosniff`, and HSTS.
- **No input sinks.** The app takes no free-text input, does no HTML
  interpolation (`innerHTML` is not used with user data), stores nothing, and
  makes no network requests.
- **Pinned CI.** GitHub Actions are pinned to full commit SHAs, workflow
  permissions are least-privilege (`contents: read` by default), and CodeQL
  scans every push and PR.
