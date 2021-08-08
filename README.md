# const_fixedp

[![docs.rs](https://docs.rs/const_fixedp/badge.svg)](https://docs.rs/const_fixedp)
[![codecov.io](https://codecov.io/gh/unleashed/const_fixedp/coverage.svg?branch=main)](https://codecov.io/gh/unleashed/const_fixedp/branch/main)
[![Build Status](https://github.com/unleashed/const_fixedp/actions/workflows/ci.yaml/badge.svg)](https://github.com/unleashed/const_fixedp/actions/workflows/CI/)
[![Security audit](https://github.com/unleashed/const_fixedp/actions/workflows/audit.yaml/badge.svg)](https://github.com/unleashed/const_fixedp/actions/workflows/Dependencies/)
[![Licensing](https://github.com/unleashed/const_fixedp/actions/workflows/license.yaml/badge.svg)](https://github.com/unleashed/const_fixedp/actions/workflows/Licensing/)
[![Clippy check](https://github.com/unleashed/const_fixedp/actions/workflows/clippy.yaml/badge.svg)](https://github.com/unleashed/const_fixedp/actions/workflows/Clippy/)
[![Rustfmt](https://github.com/unleashed/const_fixedp/actions/workflows/format.yaml/badge.svg)](https://github.com/unleashed/const_fixedp/actions/workflows/Rustfmt/)

This crate offers a basic const unsigned fixed point arithmetic type.

Please check out the documentation for examples.

## Status

This library is currently _experimental_. At this stage it is meant to explore
const-related features of rustc. If you are looking for a complete fixed point
solution you should look elsewhere in crates.io, but if you are willing to
contribute towards a better, more complete const-powered fixed point library,
then you are more than welcome!

Please report any issues at the [issue tracker](https://github.com/unleashed/const_fixedp/issues).

## MSRV

Many features that we'd wish to have available on stable Rust haven't stabilised
or landed yet. The crate itself is pretty much experimental, so there is no MSRV
policy other than "always run the latest stable version".

### Nightly compilers

It's fair game to play with nightly compiler versions implementing unstable
features _as long as_ the crate works as intended on the latest stable version.

## Contributing

Contributions are very much appreciated. This crate started out as an experiment
with generic const type parameters but ended up being potentially interesting
to others.

There are certain shortcomings, incomplete features, improvements to be done
after Rust lands support, and other tasks that anyone can do to improve the
usefulness of such a const fixed point library.

Please fork and submit pull requests!
