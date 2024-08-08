## [v1.0.0] - 2024-08-06

All notable changes to this project will be documented in this file.

### Added
- Initial release of ADI-Stable-Diffusion.
- Provide ADI command line tool and package manager installation method
- Integrated Stable Diffusion model for AI-driven image generation.
- **Feature:** Integrated and optimized Stable-Diffusion versions from v1.0 to v1.5, including the turbo version.
- **Feature:** Added support for multiple sampling methods such as Euler, Euler Ancestral, and more.
- **Feature:** Implemented Byte-Pair Encoding (BPE) and Word Piece Encoding (WP) tokenizers.
- **Feature:** Introduced a default discrete strategy for scheduling.
- **Documentation:** Added comprehensive README.md with setup instructions and usage guide.
- **CI/CD:** Integrated GitHub Actions for automated testing and deployment.

### Changed
- Update: Improved performance of the image generation algorithm.
- Update: Enhanced user interface for better user experience.
- Refactor: Codebase cleanup and modularization for better maintainability.

### Fixed
- Fixed all known issues with scheduler algorithms in the prototype version.
- Ensuring that all algorithms marked with a check in the README are now functional.
- Fixed typo in the README.md file.
- Corrected styling issues in the user dashboard.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
