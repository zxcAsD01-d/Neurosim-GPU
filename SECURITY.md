# Security policy

NeuroSim GPU is a research prototype, but it parses files and runs native code,
so security reports are welcome.

## Supported versions

| Version | Supported |
|---|---|
| `0.1.x` | yes |
| older / unreleased snapshots | no |

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability** (private security advisory).
3. Include the version or commit, a minimal reproduction, and the impact you
   expect.

If private advisories are unavailable, contact the maintainer through their
GitHub account and ask for a private channel.

We aim to acknowledge a report within a few days. This is a single-maintainer
research project, so please allow time for a fix before public disclosure.

## In scope

- Memory safety or out-of-bounds access in the CUDA/C++ extension
  (`csrc/`), including malformed topology tensors (bad row pointers, duplicate
  or out-of-range edge indices).
- Denial of service from untrusted topology or checkpoint files: for example a
  crafted `torch.save` payload, absurd graph sizes that exhaust memory during
  validation, or integer overflow in index arithmetic.
- Path or code execution issues in the helper scripts (`scripts/`,
  `benchmarks/`, `experiments/`).
- Supply-chain issues in the package metadata or CI workflows.

## Out of scope

- Scientific correctness claims, accuracy numbers and performance results. Those
  are covered by [docs/validation.md](docs/validation.md) and
  [docs/results.md](docs/results.md); open a normal issue instead.
- Vulnerabilities in PyTorch, CUDA drivers or the operating system. Report those
  upstream.
- Social engineering and physical access.

## Hardening notes for users

- Only load topology or checkpoint files you trust: they are deserialized with
  `torch.load`, which can execute arbitrary code on untrusted input.
- Run benchmarks and experiments on data you control; the MNIST download is
  fetched from the official mirror over HTTPS.
- The native extension compiles C++/CUDA at first use; build it from a source
  checkout you have reviewed.
