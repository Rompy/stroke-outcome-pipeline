# Contributing to Stroke Outcome Pipeline

Thank you for your interest in contributing! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, hardware)
- Error messages or logs

### Suggesting Enhancements

We welcome suggestions for:
- New features
- Performance improvements
- Better documentation
- Additional validation methods
- Support for other clinical domains

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/stroke-outcome-pipeline.git
   cd stroke-outcome-pipeline
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add docstrings to new functions
   - Include type hints where appropriate
   - Add tests if applicable

4. **Test your changes**
   ```bash
   # Run with synthetic data
   python scripts/generate_synthetic_data.py
   # Test your feature
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include docstrings (Google style preferred)
- Keep functions focused and modular

## Documentation

When adding new features:
- Update README.md if user-facing
- Add docstrings to all new functions
- Update configuration examples
- Add usage examples

## Testing

- Test with synthetic data before submitting
- Verify backwards compatibility
- Check memory usage for large datasets
- Ensure privacy compliance (no PHI in examples)

## Privacy Guidelines

**CRITICAL**: Never commit:
- Real patient data
- Clinical notes with PHI
- Model weights trained on patient data
- Any identifiable health information

Always use synthetic data for examples and testing.

## Questions?

- Open an issue for general questions
- Email aromchoi@yuhs.ac for private inquiries
- Check existing issues before posting

## Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Focus on improving the project
- Help others learn and grow

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Future publications (if substantial contribution)

Thank you for helping improve clinical AI research! 🏥🤖
