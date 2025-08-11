# Contributing to MegaDetector CPU Benchmark

Thank you for your interest in contributing! This guide will help you get started.

## 🤝 How to Contribute

### Reporting Issues
- Check existing issues first
- Use issue templates when available
- Include system specs (CPU, RAM, OS)
- Provide benchmark results if relevant

### Suggesting Enhancements
- Open a discussion first for major changes
- Explain the use case and benefits
- Consider backward compatibility

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🧪 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/megadetector-cpu-benchmark.git
cd megadetector-cpu-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📝 Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

### Example
```python
def calculate_fps(num_images: int, total_time: float) -> float:
    """
    Calculate frames per second.
    
    Args:
        num_images: Number of images processed
        total_time: Total processing time in seconds
        
    Returns:
        Frames per second as float
    """
    return num_images / total_time if total_time > 0 else 0
```

## 🧪 Testing

Before submitting:

1. **Run existing tests**
   ```bash
   python -m pytest tests/
   ```

2. **Add tests for new features**
   ```python
   def test_new_feature():
       assert new_feature(input) == expected_output
   ```

3. **Test Docker build**
   ```bash
   docker-compose build
   docker-compose --profile quick up
   ```

## 📊 Benchmark Contributions

When contributing optimizations:

1. **Baseline**: Record performance before changes
2. **Implement**: Make your optimization
3. **Measure**: Run full benchmark suite
4. **Document**: Include results in PR

### Benchmark Template
```markdown
## Performance Impact

### Before
- FPS: X.XX
- Latency P95: XXXms
- CPU Usage: XX%

### After  
- FPS: Y.YY (+Z%)
- Latency P95: YYYms (-Z%)
- CPU Usage: YY%

### Test Configuration
- CPU: [Model]
- RAM: [Size]
- Batch Size: [Size]
```

## 🎯 Priority Areas

### High Priority
- ARM/Apple Silicon support
- AMD CPU optimizations
- Real-time monitoring dashboard
- Kubernetes deployment configs

### Medium Priority
- Additional model support
- Cloud provider optimizations (AWS, GCP, Azure)
- Benchmark automation
- Result visualization improvements

### Low Priority
- UI/Web interface
- Database result storage
- Historical trend analysis

## 📚 Documentation

Update documentation when:
- Adding new features
- Changing configuration options
- Modifying Docker setup
- Improving performance

## 🏷️ Commit Messages

Use conventional commits:

```
feat: Add ARM processor support
fix: Correct memory calculation
docs: Update benchmark methodology
perf: Optimize batch processing
test: Add threading tests
chore: Update dependencies
```

## 🔍 Code Review Process

1. Automated checks must pass
2. At least one approval required
3. Performance impact documented
4. Tests included for new features

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 💬 Questions?

- Open a [Discussion](https://github.com/gaiar/megadetector-cpu-benchmark/discussions)
- Check existing [Issues](https://github.com/gaiar/megadetector-cpu-benchmark/issues)

Thank you for helping improve wildlife conservation technology! 🦁