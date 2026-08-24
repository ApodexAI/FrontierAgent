# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in FrontierAgent, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@apodex.ai**

Include the following in your report:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 1 week
- **Fix and disclosure**: coordinated with the reporter

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Current release |

## Scope

The following are in scope for security reports:

- Sandbox escape in `bwrap` / container isolation
- Arbitrary code execution outside the sandbox
- Credential leakage (API keys, tokens)
- Path traversal in file read/write tools
- Prompt injection leading to unauthorized actions

## Acknowledgments

We appreciate the security research community's efforts in helping keep
FrontierAgent safe. Reporters will be credited in release notes (unless
anonymity is preferred).
