# Security Policy

## Supported versions

Only the **latest release** receives security fixes. Please verify any issue
against the newest tag before reporting.

| Version   | Supported |
|-----------|-----------|
| latest release | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

- Preferred: use GitHub's
  [private vulnerability reporting](https://github.com/Windsander/ADI-Stable-Diffusion/security/advisories/new)
- Or email the maintainer: **Arikan.Li <arikanli@cyberfederal.io>**

Include: affected version/platform, reproduction steps, and the potential
impact. You can expect an acknowledgement within a few days.

## Scope notes

ADI is a local inference library/CLI that loads ONNX model files. Threats we
care about include malicious model/tokenizer files, unsafe file-path handling,
and supply-chain issues in the release chain. Model output content itself
(e.g. unsafe generations) is out of scope for this policy.
